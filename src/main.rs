#![allow(non_snake_case)]
#![feature(test)]
#![feature(const_generics)]

extern crate test;
extern crate sha3;

#[macro_use]
mod util;
mod avx_util;

mod encoding;

mod zq;
mod ring;
mod ref_r;
mod module;

mod avx_zq;
mod avx_r;

mod kyber;

use ring::*;
use module::*;
use kyber::*;

use encoding::{ Encodable, Encoder, Decoder };

use sha3::digest::{ ExtendableOutput, Input, XofReader };

use std::time::SystemTime;

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

fn cli_encrypt(key: &String, message: &String) -> encoding::Result<String>
{
    let mut key_decoder = encoding::Base64Decoder::new(key.as_str());
    let pk: PublicKey = (CompressedModule::decode(&mut key_decoder)?, key_decoder.read_bytes()?);
    let mut message_decoder = encoding::Base64Decoder::new(message.as_str());
    let message: Plaintext = message_decoder.read_bytes()?;
    let ciphertext = encrypt(&pk, message, time_seed());

    let mut encoder = encoding::Base64Encoder::new();
    ciphertext.0.encode(&mut encoder);
    ciphertext.1.encode(&mut encoder);
    return Ok(encoder.get());
}

fn cli_decrypt(key: &String, ciphertext: &String) -> encoding::Result<String>
{
    let mut key_decoder = encoding::Base64Decoder::new(key.as_str());
    let sk: SecretKey = RqVector3::decode(&mut key_decoder)?;
    let mut ciphertext_decoder = encoding::Base64Decoder::new(ciphertext.as_str());
    let ciphertext: Ciphertext = (CompressedModule::decode(&mut ciphertext_decoder)?, CompressedRq::decode(&mut ciphertext_decoder)?);
    let message: Plaintext = decrypt(&sk, ciphertext);

    let mut encoder = encoding::Base64Encoder::new();
    encoder.encode_bytes(&message);
    return Ok(encoder.get());
}

fn main() 
{
    // TODO: too long!
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
            let mut pk_encoder = encoding::Base64Encoder::new();
            pk.0.encode(&mut pk_encoder);
            pk_encoder.encode_bytes(&pk.1);
            println!("");
            println!("Public key is {}", pk_encoder.get());

            let mut sk_encoder = encoding::Base64Encoder::new();
            sk.encode(&mut sk_encoder);
            println!("");
            println!("Secret key is {}", sk_encoder.get());
            println!("");
        },
        _ => println!("Command must be one of enc, dec, gen, got command {}", args[1])
    };
}