#![allow(non_snake_case)]
#![feature(test)]
#![feature(const_generics)]
#![feature(try_trait)]

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

fn cli_encrypt(key: String, message: String) -> String
{
    let mut key_decoder = encoding::base64_decode(&key);
    let pk: PublicKey = (CompressedModule::decode(&mut key_decoder), key_decoder.read_bytes().expect("Input too short"));
    let mut message_decoder = encoding::base64_decode(&message);
    let message: Plaintext = message_decoder.read_bytes().expect("Input too short");
    let ciphertext = encrypt(&pk, message, time_seed());

    let mut result = String::new();
    {
        let mut encoder = encoding::base64_encode(&mut result);
        ciphertext.0.encode(&mut encoder);
        ciphertext.1.encode(&mut encoder);
    }
    return result;
}

fn cli_decrypt(key: String, ciphertext: String) -> String
{
    let mut key_decoder = encoding::base64_decode(&key);
    let sk: SecretKey = RqVector3::decode(&mut key_decoder);
    let mut ciphertext_decoder = encoding::base64_decode(&ciphertext);
    let ciphertext: Ciphertext = (CompressedModule::decode(&mut ciphertext_decoder), CompressedRq::decode(&mut ciphertext_decoder));
    let message: Plaintext = decrypt(&sk, ciphertext);

    let mut result = String::new();
    {
        let mut encoder = encoding::base64_encode(&mut result);
        encoder.encode_bytes(&message);
    }
    return result;
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
            let mut arguments = args.into_iter().skip(2);
            let encryption = cli_encrypt(arguments.next().unwrap(), arguments.next().unwrap());
            println!("");
            println!("Ciphertext is {}", encryption);
            println!("");
        },
        "dec" => {
            if args.len() < 4 {
                println!("Usage: crystals_kyber.exe dec secret_key ciphertext");
                return;
            }
            let mut arguments = args.into_iter().skip(2);
            let decryption = cli_decrypt(arguments.next().unwrap(), arguments.next().unwrap());
            println!("");
            println!("Plaintext is {}", decryption);
            println!("");
        },
        "gen" => {
            let (sk, pk) = key_gen(time_seed(), time_seed());
            let mut pk_string = String::new();
            {
                let mut pk_encoder = encoding::base64_encode(&mut pk_string);
                pk.0.encode(&mut pk_encoder);
                pk_encoder.encode_bytes(&pk.1);
            }
            println!("");
            println!("Public key is {}", pk_string);

            let mut sk_string = String::new();
            {
                let mut sk_encoder = encoding::base64_encode(&mut sk_string);
                sk.encode(&mut sk_encoder);
            }
            println!("");
            println!("Secret key is {}", sk_string);
            println!("");
        },
        _ => println!("Command must be one of enc, dec, gen, got command {}", args[1])
    };
}