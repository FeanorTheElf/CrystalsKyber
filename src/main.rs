#![allow(non_snake_case)]
#![feature(test)]
#![feature(const_generics)]

extern crate test;
extern crate sha3;

#[macro_use]
mod util;
mod avx_util;

mod base64;

mod zq;
mod ring;
mod module;

mod ref_r;
mod avx_zq;
mod avx_r;

use zq::*;
use ring::*;
use module::*;

use base64::Encodable;

use sha3::digest::{ ExtendableOutput, Input, XofReader };

use std::time::SystemTime;

type Rq = avx_r::Rq;
type FourierReprRq = avx_r::FourierReprRq;
type Module = module::Module<Rq>;
type Matrix = module::Matrix<Rq>;

const COMPRESSION_VECTOR: u16 = 11;
const COMPRESSION_ELEMENT: u16 = 3;

type Seed = [u8; 32];
type PublicKey = (CompressedM<COMPRESSION_VECTOR>, Seed);
type SecretKey = Module;
type Ciphertext = (CompressedM<COMPRESSION_VECTOR>, CompressedRq<COMPRESSION_ELEMENT>);
type Plaintext = [u8; 32];

fn encrypt(pk: &PublicKey, plaintext: Plaintext, enc_seed: Seed) -> Ciphertext
{
    let mut noise_random = expand_randomness_shake_256(enc_seed);
    let r = sample_error_distribution_vector(&mut noise_random);
    let e1 = sample_error_distribution_vector(&mut noise_random);
    let e2 = sample_error_distribution_element(&mut noise_random).dft();
    let A = sample_uniform_matrix(expand_randomness_shake_128(pk.1));
    let u = A.transpose() * &r + &e1;
    let t = Module::decompress(&pk.0);
    let message = Rq::decompress(&CompressedRq::from_data(plaintext));
    let v = (&t * &r) + &e2 + &message.dft(); 
    return (u.compress(), v.inv_dft().compress());
}

fn decrypt(sk: &SecretKey, c: Ciphertext) -> Plaintext
{
    let m = Rq::decompress(&c.1) - &(sk * &Module::decompress(&c.0)).inv_dft();
    return m.compress().get_data();
}

fn key_gen(matrix_seed: Seed, secret_seed: Seed) -> (SecretKey, PublicKey)
{
    let a: Matrix = sample_uniform_matrix(expand_randomness_shake_128(matrix_seed));
    let mut noise = expand_randomness_shake_256(secret_seed);
    let s: Module = sample_error_distribution_vector(&mut noise);
    let e: Module = sample_error_distribution_vector(&mut noise);
    let b: Module = &a * &s + &e;
    return (s, (b.compress(), matrix_seed));
}

fn sample_uniform_zq<T: XofReader>(mut reader: T) -> impl Iterator<Item = Zq>
{
    let mut buffer: [u8; 2] = [0, 0];
    std::iter::repeat(()).map(move |_| {
        loop {
            reader.read(&mut buffer);
            let val: i16 = unsafe { std::mem::transmute::<_, i16>(buffer) } & 0x1FFF;
            debug_assert!(val >= 0);
            if val < Q as i16 {
                return Zq::from_perfect(val);
            }
        }
    })
}

fn sample_centered_binomial_distribution(random: u8) -> Zq
{
    let mut value = (random << 4).count_ones() as i16 - (random >> 4).count_ones() as i16;
    if value < 0 {
        value += Q as i16;
    }
    return Zq::from_perfect(value);
}

fn sample_error_distribution_vector<T: XofReader>(reader: &mut T) -> Module
{
    let mut buffer: [u8; N] = [0; N];
    let data = util::create_array(|_| {
        reader.read(&mut buffer);
        Rq::from(util::create_array(|i| sample_centered_binomial_distribution(buffer[i]))).dft()
    });
    return Module::from(data);
}

fn sample_error_distribution_element<T: XofReader>(reader: &mut T) -> Rq
{
    let mut buffer: [u8; N] = [0; N];
    reader.read(&mut buffer);
    return Rq::from(util::create_array(|i| sample_centered_binomial_distribution(buffer[i])));
}

fn expand_randomness_shake_256(seed: Seed) -> sha3::Sha3XofReader
{
    let mut hasher = sha3::Shake256::default();
    hasher.input(&seed);
    return hasher.xof_result();
}

fn expand_randomness_shake_128(seed: Seed) -> sha3::Sha3XofReader
{
    let mut hasher = sha3::Shake128::default();
    hasher.input(&seed);
    return hasher.xof_result();
}

fn sample_uniform_matrix<T: XofReader>(reader: T) -> Matrix
{
    let mut iter = sample_uniform_zq(reader);
    let data: [[FourierReprRq; DIM]; DIM] = util::create_array(|_row| 
        util::create_array(|_col| {
            Rq::from(util::create_array_it(&mut iter)).dft()
        })
    );
    return Matrix::from(data);
}

fn time_seed() -> Seed
{
    let nanos: u32 = (SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() % 1073741824) as u32;
    let mut hasher = sha3::Shake256::default();
    let data: [u8; 4] = util::create_array(|i| ((nanos >> (i * 8)) & 0xFF) as u8);
    hasher.input(&data);
    let mut result = [0; 32];
    hasher.xof_result().read(&mut result);
    return result;
}

fn cli_encrypt(key: &String, message: &String) -> base64::Result<String>
{
    let mut key_decoder = base64::Decoder::new(key.as_str());
    let pk: PublicKey = (CompressedM::decode(&mut key_decoder)?, key_decoder.read_bytes()?);
    let mut message_decoder = base64::Decoder::new(message.as_str());
    let message: Plaintext = message_decoder.read_bytes()?;
    let ciphertext = encrypt(&pk, message, time_seed());

    let mut encoder = base64::Encoder::new();
    ciphertext.0.encode(&mut encoder);
    ciphertext.1.encode(&mut encoder);
    return Ok(encoder.get());
}

fn cli_decrypt(key: &String, ciphertext: &String) -> base64::Result<String>
{
    let mut key_decoder = base64::Decoder::new(key.as_str());
    let sk: SecretKey = Module::decode(&mut key_decoder)?;
    let mut ciphertext_decoder = base64::Decoder::new(ciphertext.as_str());
    let ciphertext: Ciphertext = (CompressedM::decode(&mut ciphertext_decoder)?, CompressedRq::decode(&mut ciphertext_decoder)?);
    let message: Plaintext = decrypt(&sk, ciphertext);

    let mut encoder = base64::Encoder::new();
    encoder.encode_bytes(&message);
    return Ok(encoder.get());
}

fn main() 
{
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2{
        println!("Usage: crystals_kyber.exe command parameters...");
        return;
    }
    match args[1].as_str() {
        "enc" => {
            if args.len() < 4 {
                println!("Usage: crystals_kyber.exe enc public_key plaintext");
                println!("  where message are 32 base64 encoded bytes, i.e. 43 characters of A-Z, a-z, 0-9, +, / followed by =");
                println!("  longer messages are also allowed, then only the prefix will be used");
                return;
            }
            let encryption = cli_encrypt(&args[2], &args[3]);
            match encryption {
                Ok(ciphertext) => {
                    println!("");
                    println!("Ciphertext is {}", ciphertext);
                    println!("");
                },
                Err(_) => {
                    println!("Message or key invalid!");
                }
            }
        },
        "dec" => {
            if args.len() < 4 {
                println!("Usage: crystals_kyber.exe dec secret_key ciphertext");
                return;
            }
            let decryption = cli_decrypt(&args[2], &args[3]);
            match decryption {
                Ok(plaintext) => {
                    println!("");
                    println!("Plaintext is {}", plaintext);
                    println!("");
                },
                Err(_) => {
                    println!("Ciphertext or key invalid!");
                }
            }
        },
        "gen" => {
            let (sk, pk) = key_gen(time_seed(), time_seed());
            let mut pk_encoder = base64::Encoder::new();
            pk.0.encode(&mut pk_encoder);
            pk_encoder.encode_bytes(&pk.1);
            println!("");
            println!("Public key is {}", pk_encoder.get());

            let mut sk_encoder = base64::Encoder::new();
            sk.encode(&mut sk_encoder);
            println!("");
            println!("Secret key is {}", sk_encoder.get());
            println!("");
        },
        _ => println!("Command must be one of enc, dec, gen, got command {}", args[1])
    };
}

#[cfg(test)]
const TEST_MESSAGE: Plaintext = [
    0x00, 0x01, 0xFA, 0x09, 0x53, 0xFF, 0xF0, 0x38, 0x19, 0xA4, 0x4D, 0x82, 0x28, 0x64, 0xEF, 0x00, 
    0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
];

#[bench]
fn bench_all(bencher: &mut test::Bencher) {
    bencher.iter(|| {
        let mut matrix_seed = [0; 32];
        matrix_seed[0] = 1;
        let mut secret_seed = [0; 32];
        secret_seed[1] = 2;
        let mut enc_seed = [0; 32];
        enc_seed[2] = 3;

        let (sk, pk) = key_gen(matrix_seed, secret_seed);
        let ciphertext = encrypt(&pk, TEST_MESSAGE, enc_seed);
        let message = decrypt(&sk, ciphertext);
        for j in 0..32 {
            assert!(TEST_MESSAGE[j] == message[j], "Expected messages to be the same, differ at index {}: {} != {}", j, TEST_MESSAGE[j], message[j]);
        }
    });
}

#[bench]
fn bench_key_generation(bencher: &mut test::Bencher) {
    bencher.iter(|| {
        let mut matrix_seed = [0; 32];
        matrix_seed[0] = 1;
        let mut secret_seed = [0; 32];
        secret_seed[1] = 2;

        let (_sk, _pk) = key_gen(matrix_seed, secret_seed);
    });
}

#[bench]
fn bench_encryption(bencher: &mut test::Bencher) {
    let mut matrix_seed = [0; 32];
    matrix_seed[0] = 1;
    let mut secret_seed = [0; 32];
    secret_seed[1] = 2;
    let mut enc_seed = [0; 32];
    enc_seed[2] = 3;

    let (_sk, pk) = key_gen(matrix_seed, secret_seed);
    bencher.iter(|| {
        let _ciphertext = encrypt(&pk, TEST_MESSAGE, enc_seed);
    });
}

#[bench]
fn bench_decryption(bencher: &mut test::Bencher) {
    let mut matrix_seed = [0; 32];
    matrix_seed[0] = 1;
    let mut secret_seed = [0; 32];
    secret_seed[1] = 2;
    let mut enc_seed = [0; 32];
    enc_seed[2] = 3;

    let (sk, pk) = key_gen(matrix_seed, secret_seed);
    let ciphertext = encrypt(&pk, TEST_MESSAGE, enc_seed);
    bencher.iter(|| {
        let _m = decrypt(&sk, ciphertext.clone());
    });
}