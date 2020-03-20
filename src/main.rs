#![allow(dead_code)]
#![feature(test)]
#![feature(const_generics)]

extern crate test;
extern crate sha3;

#[macro_use]
mod util;
mod avx_util;

mod base64;

mod zq;
mod r;
mod m;

mod ref_r;
mod avx_zq;
mod avx_r;

use zq::*;
use r::*;
use m::*;

use sha3::digest::{ ExtendableOutput, Input, XofReader };

use std::time::SystemTime;

type R = avx_r::R;
type FourierReprR = avx_r::FourierReprR;
type M = m::Module<R>;
type Mat = m::Mat<R>;

const COMPRESSION_VECTOR: u16 = 11;
const COMPRESSION_ELEMENT: u16 = 3;

type PK = (CompressedM<COMPRESSION_VECTOR>, [u8; 32]);
type SK = M;
type Ciphertext = (CompressedM<COMPRESSION_VECTOR>, CompressedR<COMPRESSION_ELEMENT>);
type Message = [u8; 32];

fn enc(pk: &PK, plaintext: Message, enc_seed: [u8; 32]) -> Ciphertext
{
    let mut noise = expand_randomness_shake_256(&enc_seed);
    let r = sample_error_distribution_vector(&mut noise);
    let e1 = sample_error_distribution_vector(&mut noise);
    let e2 = sample_error_distribution_element(&mut noise).dft();
    let a = sample_uniform_matrix(expand_randomness_shake_128(&pk.1));
    let u = a.transpose() * &r + &e1;
    let t = M::decompress(&pk.0);
    let message = R::decompress(&CompressedR::from_data(plaintext));
    let v = (&t * &r) + &e2 + &message.dft(); 
    return (u.compress(), v.inv_dft().compress());
}

fn dec(sk: &SK, c: Ciphertext) -> Message
{
    let m = R::decompress(&c.1) - &(sk * &M::decompress(&c.0)).inv_dft();
    return m.compress().get_data();
}

fn key_gen(matrix_seed: [u8; 32], secret_seed: [u8; 32]) -> (SK, PK)
{
    let a: Mat = sample_uniform_matrix(expand_randomness_shake_128(&matrix_seed));
    let mut noise = expand_randomness_shake_256(&secret_seed);
    let s: M = sample_error_distribution_vector(&mut noise);
    let e: M = sample_error_distribution_vector(&mut noise);
    let b: M = &a * &s + &e;
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

fn sample_error_distribution_vector<T: XofReader>(reader: &mut T) -> M
{
    let mut buffer: [u8; N] = [0; N];
    let data = util::create_array(|_| {
        reader.read(&mut buffer);
        R::from(util::create_array(|i| sample_centered_binomial_distribution(buffer[i]))).dft()
    });
    return M::from(data);
}

fn sample_error_distribution_element<T: XofReader>(reader: &mut T) -> R
{
    let mut buffer: [u8; N] = [0; N];
    reader.read(&mut buffer);
    return R::from(util::create_array(|i| sample_centered_binomial_distribution(buffer[i])));
}

fn expand_randomness_shake_256(seed: &[u8; 32]) -> sha3::Sha3XofReader
{
    let mut hasher = sha3::Shake256::default();
    hasher.input(&seed);
    return hasher.xof_result();
}

fn expand_randomness_shake_128(seed: &[u8; 32]) -> sha3::Sha3XofReader
{
    let mut hasher = sha3::Shake128::default();
    hasher.input(&seed);
    return hasher.xof_result();
}

fn sample_uniform_matrix<T: XofReader>(reader: T) -> Mat
{
    let mut iter = sample_uniform_zq(reader);
    let data: [[FourierReprR; DIM]; DIM] = util::create_array(|_row| 
        util::create_array(|_col| {
            R::from(util::create_array_it(&mut iter)).dft()
        })
    );
    return Mat::from(data);
}

fn time_seed() -> [u8; 32]
{
    let nanos: u32 = (SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() % 1073741824) as u32;
    let mut hasher = sha3::Shake256::default();
    let data: [u8; 4] = util::create_array(|i| ((nanos >> (i * 8)) & 0xFF) as u8);
    hasher.input(&data);
    let mut result = [0; 32];
    hasher.xof_result().read(&mut result);
    return result;
}

fn main() 
{
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2{
        println!("Usage: crystals_kyber.exe command parameters... [options...]");
        return;
    }
    match args[1].as_str() {
        "enc" => {
            if args.len() < 4 {
                println!("Usage: crystals_kyber.exe enc public_key message [options...]");
                println!("  where message are 32 base64 encoded bytes");
                return;
            }
            let mut key_decoder = base64::Decoder::new(args[2].as_str());
            let pk: PK = (CompressedM::decode(&mut key_decoder), key_decoder.read_bytes());
            let mut message_decoder = base64::Decoder::new(args[3].as_str());
            let message: Message = message_decoder.read_bytes();
            let ciphertext = enc(&pk, message, time_seed());

            let mut encoder = base64::Encoder::new();
            ciphertext.0.encode(&mut encoder);
            ciphertext.1.encode(&mut encoder);
            println!("Ciphertext is {}", encoder.get());
        },
        "dec" => {
            if args.len() < 4 {
                println!("Usage: crystals_kyber.exe dec secret_key ciphertext [options...]");
                return;
            }
            let mut key_decoder = base64::Decoder::new(args[2].as_str());
            let sk: SK = M::decode(&mut key_decoder);
            let mut ciphertext_decoder = base64::Decoder::new(args[3].as_str());
            let ciphertext: Ciphertext = (CompressedM::decode(&mut ciphertext_decoder), CompressedR::decode(&mut ciphertext_decoder));
            let message: Message = dec(&sk, ciphertext);

            let mut encoder = base64::Encoder::new();
            encoder.encode_bytes(&message);
            println!("Message is {}", encoder.get());
        },
        "gen" => {
            let (sk, pk) = key_gen(time_seed(), time_seed());
            let mut pk_encoder = base64::Encoder::new();
            pk.0.encode(&mut pk_encoder);
            pk_encoder.encode_bytes(&pk.1);
            println!("Public key is {}", pk_encoder.get());

            let mut sk_encoder = base64::Encoder::new();
            sk.encode(&mut sk_encoder);
            println!("Secret key is {}", sk_encoder.get());
        },
        _ => println!("Command must be one of enc, dec, gen, got command {}", args[1])
    };
}

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
        let mut expected_message: Message = [0; 32];
        expected_message[0] = 1;
        expected_message[2] = 0xF0;
        expected_message[12] = 0x1A;
        let ciphertext = enc(&pk, expected_message, enc_seed);
        let message = dec(&sk, ciphertext);
        for j in 0..32 {
            assert!(expected_message[j] == message[j], "Expected messages to be the same, differ at index {}: {} != {}", j, expected_message[j], message[j]);
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
    let mut expected_message: Message = [0; 32];
    expected_message[0] = 1;
    expected_message[2] = 0xF0;
    expected_message[12] = 0x1A;
    bencher.iter(|| {
        let _ciphertext = enc(&pk, expected_message, enc_seed);
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
    let mut expected_message: Message = [0; 32];
    expected_message[0] = 1;
    expected_message[2] = 0xF0;
    expected_message[12] = 0x1A;
    let ciphertext = enc(&pk, expected_message, enc_seed);
    bencher.iter(|| {
        let _m = dec(&sk, ciphertext.clone());
    });
}