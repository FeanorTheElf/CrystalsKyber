use super::util;

#[cfg(not(target_feature = "avx2"))]
use super::ref_r;
#[cfg(target_feature = "avx2")]
use super::avx_r;

use super::rqvec;
use super::zq::*;
use super::rqvec::*;
use super::ring::{ RqElementChineseRemainderRepr, RqElementCoefficientRepr, CompressedRq, N };

use sha3::digest::{ ExtendableOutput, Input, XofReader };

#[cfg(not(target_feature = "avx2"))]
pub type RqElement = ref_r::RqElementCoefficientReprImpl;
#[cfg(target_feature = "avx2")]
pub type RqElement = avx_r::RqElementCoefficientReprImpl;

pub type RqVector = rqvec::RqVector3<RqElement>;
pub type RqMatrix = rqvec::RqSquareMatrix3<RqElement>;

/// Each public key and ciphertext element is compressed using this count of bits
const COMPRESSED_VECTOR_BIT_SIZE: u16 = 11;
const COMPRESSED_RING_ELEMENT_BIT_SIZE: u16 = 3;

pub type Seed = [u8; 32];
pub type PublicKey = (CompressedRqVector<COMPRESSED_VECTOR_BIT_SIZE>, Seed);
pub type SecretKey = RqVector;
pub type Ciphertext = (CompressedRqVector<COMPRESSED_VECTOR_BIT_SIZE>, CompressedRq<COMPRESSED_RING_ELEMENT_BIT_SIZE>);
pub type Plaintext = [u8; 32];

pub fn encrypt(pk: &PublicKey, plaintext: Plaintext, enc_seed: Seed) -> Ciphertext
{
    let t = RqVector::decompress(&pk.0);
    let A = sample_uniform_matrix(&pk.1);
    let mut noise_random = expand_randomness_shake_256(enc_seed);
    let r = sample_error_distribution_vector(&mut noise_random);
    let e1 = sample_error_distribution_vector(&mut noise_random);
    let e2 = sample_error_distribution_element(&mut noise_random);
    let u = A.transpose() * &r + &e1;
    let message = RqElement::decompress(&CompressedRq::from_data(plaintext));
    let v = (&t * &r).to_coefficient_repr() + &e2 + &message;
    return (u.compress(), v.compress());
}

pub fn decrypt(sk: SecretKey, c: Ciphertext) -> Plaintext
{
    let u = RqVector::decompress(&c.0);
    let v = RqElement::decompress(&c.1);
    let m = v - &(&sk * &u).to_coefficient_repr();
    return m.compress().get_data();
}

pub fn key_gen(matrix_seed: Seed, secret_seed: Seed) -> (SecretKey, PublicKey)
{
    let A: RqMatrix = sample_uniform_matrix(&matrix_seed);
    let mut noise = expand_randomness_shake_256(secret_seed);
    let s: RqVector = sample_error_distribution_vector(&mut noise);
    let e: RqVector = sample_error_distribution_vector(&mut noise);
    let b: RqVector = &A * &s + &e;
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

fn sample_uniform_matrix(seed: &Seed) -> RqMatrix
{
    let mut buffer = [0; 34];
    for k in 0..32 {
        buffer[k] = seed[k];
    }
    RqMatrix::from(util::create_array(|row|
        util::create_array(|col|{
            let mut hasher = sha3::Shake128::default();
            buffer[32] = col as u8;
            buffer[33] = row as u8;
            hasher.input(&buffer[..]);
            let mut reader = sample_uniform_zq(hasher.xof_result());
            <RqElement as RqElementCoefficientRepr>::ChineseRemainderRepr::from(
                util::create_array_it(&mut reader)
            )
        })
    ))
}

fn sample_centered_binomial_distribution(random: u8) -> ZqElement
{
    let mut value = (random << 4).count_ones() as i16 - (random >> 4).count_ones() as i16;
    if value < 0 {
        value += Q as i16;
    }
    return ZqElement::from_perfect(value);
}

fn sample_error_distribution_vector<T: XofReader>(reader: &mut T) -> RqVector
{
    let data = util::create_array(|_| {
        sample_error_distribution_element(reader).to_chinese_remainder_repr()
    });
    return RqVector::from(data);
}

fn sample_error_distribution_element<T: XofReader>(reader: &mut T) -> RqElement
{
    let mut buffer: [u8; N] = [0; N];
    reader.read(&mut buffer);
    return RqElement::from(util::create_array(|i| 
        sample_centered_binomial_distribution(buffer[i])
    ));
}

fn expand_randomness_shake_256(seed: Seed) -> sha3::Sha3XofReader
{
    let mut hasher = sha3::Shake256::default();
    hasher.input(&seed);
    return hasher.xof_result();
}

#[cfg(test)]
const TEST_MESSAGE: Plaintext = [
    0x00, 0x01, 0xFA, 0x09, 0x53, 0xFF, 0xF0, 0x38, 0x19, 0xA4, 0x4D, 0x82, 0x28, 0x64, 0xEF, 0x00, 
    0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
];

#[cfg(test)]
const TEST_SEED: Seed = [
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00
];

#[test]
fn test_roundtrip() 
{
    let (sk, pk) = key_gen(TEST_SEED, TEST_SEED);
    let ciphertext = encrypt(&pk, TEST_MESSAGE, TEST_SEED);
    let message = decrypt(sk, ciphertext);
    assert_eq!(TEST_MESSAGE, message);
}

#[bench]
fn benchmark_all(bencher: &mut test::Bencher) 
{
    bencher.iter(|| {
        let (sk, pk) = key_gen(TEST_SEED, TEST_SEED);
        let ciphertext = encrypt(&pk, TEST_MESSAGE, TEST_SEED);
        let message = decrypt(sk, ciphertext);
        std::hint::black_box(message);
    });
}

#[bench]
fn benchmark_key_generation(bencher: &mut test::Bencher)
{
    bencher.iter(|| {
        let (sk, pk) = key_gen(TEST_SEED, TEST_SEED);
        std::hint::black_box(sk);
        std::hint::black_box(pk);
    });
}

#[bench]
fn benchmark_encryption(bencher: &mut test::Bencher) 
{
    let (_sk, pk) = key_gen(TEST_SEED, TEST_SEED);

    bencher.iter(|| {
        let ciphertext = encrypt(&pk, TEST_MESSAGE, TEST_SEED);
        std::hint::black_box(ciphertext)
    });
}

#[bench]
fn benchmark_decryption(bencher: &mut test::Bencher) 
{
    let (sk, pk) = key_gen(TEST_SEED, TEST_SEED);
    let ciphertext = encrypt(&pk, TEST_MESSAGE, TEST_SEED);

    bencher.iter(|| {
        let message = decrypt(sk.clone(), ciphertext.clone());
        std::hint::black_box(message);
    });
}