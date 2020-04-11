use super::util;
use super::ref_r;
use super::avx_r;
use super::module;
use super::zq::*;
use super::module::*;
use super::ring::*;

use sha3::digest::{ ExtendableOutput, Input, XofReader };

type RqElement = ref_r::RqElementCoefficientReprImpl;
type Module = module::RqVector3<RqElement>;
type Matrix = module::RqSquareMatrix3<RqElement>;

/// Each public key and ciphertext element is compressed using this count of bits
const COMPRESSED_VECTOR_BIT_SIZE: u16 = 11;
const COMPRESSED_RING_ELEMENT_BIT_SIZE: u16 = 3;

pub type Seed = [u8; 32];
pub type PublicKey = (CompressedModule<COMPRESSED_VECTOR_BIT_SIZE>, Seed);
pub type SecretKey = Module;
pub type Ciphertext = (CompressedModule<COMPRESSED_VECTOR_BIT_SIZE>, CompressedRq<COMPRESSED_RING_ELEMENT_BIT_SIZE>);
pub type Plaintext = [u8; 32];

pub fn encrypt(pk: &PublicKey, plaintext: Plaintext, enc_seed: Seed) -> Ciphertext
{
    let t = Module::decompress(&pk.0);
    let A = sample_uniform_matrix(expand_randomness_shake_128(pk.1));
    let mut noise_random = expand_randomness_shake_256(enc_seed);
    let r = sample_error_distribution_vector(&mut noise_random);
    let e1 = sample_error_distribution_vector(&mut noise_random);
    let e2 = sample_error_distribution_element(&mut noise_random).to_chinese_remainder_repr();
    let u = A.transpose() * &r + &e1;
    let message = RqElement::decompress(&CompressedRq::from_data(plaintext));
    let v = (&t * &r) + &e2 + &message.to_chinese_remainder_repr(); 
    return (u.compress(), v.to_coefficient_repr().compress());
}

pub fn decrypt(sk: &SecretKey, c: Ciphertext) -> Plaintext
{
    let u = Module::decompress(&c.0);
    let v = RqElement::decompress(&c.1);
    let m = v - &(sk * &u).to_coefficient_repr();
    return m.compress().get_data();
}

pub fn key_gen(matrix_seed: Seed, secret_seed: Seed) -> (SecretKey, PublicKey)
{
    let a: Matrix = sample_uniform_matrix(expand_randomness_shake_128(matrix_seed));
    let mut noise = expand_randomness_shake_256(secret_seed);
    let s: Module = sample_error_distribution_vector(&mut noise);
    let e: Module = sample_error_distribution_vector(&mut noise);
    let b: Module = &a * &s + &e;
    return (s, (b.compress(), matrix_seed));
}

fn sample_uniform_zq<T: XofReader>(mut reader: T) -> impl Iterator<Item = ZqElement>
{
    let mut buffer: [u8; 2] = [0, 0];
    std::iter::repeat(()).map(move |_| {
        loop {
            reader.read(&mut buffer);
            let val: i16 = unsafe { std::mem::transmute::<_, i16>(buffer) } & 0x1FFF;
            debug_assert!(val >= 0);
            if val < Q as i16 {
                return ZqElement::from_perfect(val);
            }
        }
    })
}

fn sample_uniform_matrix<T: XofReader>(reader: T) -> Matrix
{
    let mut iter = sample_uniform_zq(reader);
    let matrix_content = util::create_array(|_row| 
        util::create_array(|_col| {
            RqElement::from(util::create_array_it(&mut iter)).to_chinese_remainder_repr()
        })
    );
    return Matrix::from(matrix_content);
}

fn sample_centered_binomial_distribution(random: u8) -> ZqElement
{
    let mut value = (random << 4).count_ones() as i16 - (random >> 4).count_ones() as i16;
    if value < 0 {
        value += Q as i16;
    }
    return ZqElement::from_perfect(value);
}

fn sample_error_distribution_vector<T: XofReader>(reader: &mut T) -> Module
{
    let mut buffer: [u8; N] = [0; N];
    let data = util::create_array(|_| {
        reader.read(&mut buffer);
        RqElement::from(util::create_array(|i| sample_centered_binomial_distribution(buffer[i]))).to_chinese_remainder_repr()
    });
    return Module::from(data);
}

fn sample_error_distribution_element<T: XofReader>(reader: &mut T) -> RqElement
{
    let mut buffer: [u8; N] = [0; N];
    reader.read(&mut buffer);
    return RqElement::from(util::create_array(|i| sample_centered_binomial_distribution(buffer[i])));
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


#[cfg(test)]
const TEST_MESSAGE: Plaintext = [
    0x00, 0x01, 0xFA, 0x09, 0x53, 0xFF, 0xF0, 0x38, 0x19, 0xA4, 0x4D, 0x82, 0x28, 0x64, 0xEF, 0x00, 
    0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
];

#[bench]
fn benchmark_all(bencher: &mut test::Bencher) {
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
fn benchmark_key_generation(bencher: &mut test::Bencher) {
    let mut matrix_seed = [0; 32];
    matrix_seed[0] = 1;
    let mut secret_seed = [0; 32];
    secret_seed[1] = 2;
    
    bencher.iter(|| {
        let (_sk, _pk) = key_gen(matrix_seed, secret_seed);
    });
}

#[bench]
fn benchmark_encryption(bencher: &mut test::Bencher) {
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
fn benchmark_decryption(bencher: &mut test::Bencher) {
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