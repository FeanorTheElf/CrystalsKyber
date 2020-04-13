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
mod rqvec;

mod avx_zq;
mod avx_r;

mod kyber;

mod ref_impl_compat;

use kyber::*;

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

fn cli_encrypt(key: &str, message: &str) -> String
{
    let pk: PublicKey = ref_impl_compat::read_pk_from_ref_impl(key);
    let message: Plaintext = ref_impl_compat::read_message_from_ref_impl(message);
    let ciphertext = encrypt(&pk, message, time_seed());
    return ref_impl_compat::write_ciphertext_to_ref_impl(&ciphertext);
}

fn cli_decrypt(key: &str, ciphertext: &str) -> String
{
    let ciphertext: Ciphertext = ref_impl_compat::read_ciphertext_from_ref_impl(ciphertext);
    let sk: SecretKey = ref_impl_compat::read_sk_from_ref_impl(key);
    let message: Plaintext = decrypt(&sk, ciphertext);
    return ref_impl_compat::write_message_to_ref_impl(&message);
}

fn cli_key_gen() -> (String, String)
{
    let pk_seed = time_seed();
    let mut sk_seed = time_seed();
    sk_seed[0] ^= 0xF;
    let (sk, pk) = key_gen(pk_seed, sk_seed);
    return (ref_impl_compat::write_sk_to_ref_impl(&sk), ref_impl_compat::write_pk_to_ref_impl(&pk));
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
            println!("");
            println!("Ciphertext is {}", encryption);
            println!("");
        },
        "dec" => {
            if args.len() < 4 {
                println!("Usage: crystals_kyber.exe dec secret_key ciphertext");
                return;
            }
            let decryption = cli_decrypt(&args[2], &args[3]);
            println!("");
            println!("Plaintext is {}", decryption);
            println!("");
        },
        "gen" => {
            let (sk, pk) = cli_key_gen();
            println!("");
            println!("Public key is {}", pk);

            println!("");
            println!("Secret key is {}", sk);
            println!("");
        },
        _ => println!("Command must be one of enc, dec, gen, got command {}", args[1])
    };
}

#[test]
fn test_interaction_reference_implementation() {
    let ciphertext_str = "\
        Kv4Iun+HlotqPIJ5pCXaRKIQSZOPTbNOWdOxikoKkopYML2+rPbRMzzNwx6SJB65hWHuHS2wunQ+gR9r0djMM9Hw8D3uldWIAiC3zNN8cVYmPuu7+IqmFdtSI7E8ypJfWGQz\
        COGd+2Y3PaOfmC+lDld7PYck5XRQ3KLfAtL0vc1GmCSnmjziFNEKm9k7w/AO6qIr4rB6GMcS7IoMvXVj7StMx0D4DKgIQ82f89NBf1sOobKhXQTJaXka4+2/A4/bATDa2+iP\
        oOXqzBxQcFttWOW5Wjh/0VJqs9/BWeICpJA0yhXWX7631Sw0WgAFkxbPTxeV/coTdyvcu/3QKpEkONrW5NYWN9cDwuibFZhA6WcFfbU+98rmcWPWiikL7IMRj8Yv8ZL/jRV6\
        9ubugGWb/8tvpYlRCrjwzzQJRbVAsgepHEPulSnH+sojinA2F3F1wM+0ck9KaH0kImkbm8eAn2owBIsG6zO31VJC2A8xLPIX7LeIglPVzIlLxJJ5+m7sqb142R/mQ2GhAYbc\
        1iRrALvBJbEXgFi/Rz5fLMd/JGsT1MEnLTcNs1ojIh/jBGFsoFN3k9Ir7SI33uItyJ6C9IXWwPq2REIf4CsBDPDfuwGg/TskEZLEEHwAj0UG4Vp9kXBIA1eD35+YHLvTGaK9\
        MMoVQ+iHVDZOZCTrobsLbIpk9/xWNGTUJ+FiMb6S25S0bTxCYvzZ6OFtIDMgsso5oVunFu1CBItm6Wv3iG7v2msY4EniwbnaYOEw9E47yc0ky3DAao+MaTzhLgSl/9JFvw5g\
        A6q9p9qp0ysCQ6hV0u6cdPz8QJZGCighXkPTzSe5NL2OIt9XCOPV3Lvw/uaCUtMkJuUtWKOO4hp+Rg2Zg9PvM8uHFLeNkxujtARo3JYmIcgd3mJGxW5vqiIIGMYwGGboGj94\
        HMKCsNFipnhmmRra+eHx+9NKkH6idb0gWc5rCziCEobKSKEeEurRuTGcHabz7Msi8Gbk25xCwp2K2tNfvYhjoGSS1Yvo8ZS+NADnOvv93HU6in4OjZic5Gdatr/7Db0UFbQn\
        gqsjMDZdcd2iBGOmad8w1p3tMHnp7KtqqH2DgHLAhkwo0qWwNsNP2vFmXjxvWSlbYoJ5QfTLdlCyKyMxuPzBVHCywJNni6O+0cE4SVU4zfod4qu9WZCPWFJ5WVRaxrmBrB0z\
        c0QzqUr5oTv2TOd4CYGEzUS3VCGHVKRuzHzvCZtstid8sfw+g333g4ZXH1LczacZZJQbouGlhcXCW3B/Xnokjc2PmR6675Gfo+EBgC5bzwrDKTnP2nGh0r91pXqJnivOlOva\
        pSejPrqgarishjJHi0Sky802uIC2aNz5wb5DphUHrTibMrfEoiepEsdMGNQBy6yZI5lnNNEcxbBtm5hB190oVBhJtcAT7ILdtHpu1gt02XFd1wO9HWA9O/46h7rFXGJ2Q+HM\
        ZjB++kxiGOjdj5bKDZARXM3ImEXSIGGevkebkhOhI6o+envMgWfymSKgaUQIhb9PAZbidzN9H5nKFh7in+3i";
    let sk_str = "\
        uCaLtxyAWACnccmLQx9WvQDjgD1e6IpfOqON+aFeqhVAeJejyfVTa4trYb/r041+VmidMlPLsZqSeHTQnlDHsAdlpg/I6MQZdduv+WgAfer4IQcrSzOQnYwRQ1J0oHsWcPa1Yh8ml3\
        PH8qlJ3wVpEo3OyWkndYiSgt9hB/4e0hlge37UyDoVfd0NQ65FuYwBbq/2Q5AnncHFJYnuOw52OBXzQCf7uGvU5JgGZC6QVPRmK1fqvu8XpUaOODD1m5eOybR4a6Aa9Q5o4Dzb2VXo\
        G61OvL81sX5OURhYV+4EDN+8jyEBuxtNIxReweuaM1N3vXCEBehpZv7tla7CMxxe4jZ0FNwBjGXpmthQQdo0v50IpQZghKk3wVtpo0z4YRIdoKnnEM6FrVDMRo7msqBkhRVW0fBW98\
        JFVBk/pr7grKQOEZnYz1vuDsejyVSV5p7R7sQJInPHtilnC/aGikOlSXeMHYDGC6HD8LeeApuszIUga5T+uBD1U1hXRe3EcixClvjbB+4afDe4gxCugw+Dmdr+WrUuemqRR5C6gGK2\
        NmTYUXN2EImsGcT2SUXd2/q14J/Kicjvgu+sIsMZh/f2V4cGP+wWNodMxBOwAh0avkPalQDhi+CMh0OPpGybUMMtVPv5DA9p5I5flQQs8SQAnBna5XydTTr35SNPIWkq7xRJhhuo+3\
        R1qsBHQtwb1R6AG+e7HgywUmD+3UVuuV7ZsOKffLJSHmBeeMVL0l5TiKwHlq5BSg1GRQFYka74/YyVWX2WP0BCJ7Q2hM+wE7y+yKNrBPYcP0dYdoO7payaYle5QoDbLgXPYOcShxjU\
        NV4Zh/UUC+0C8LigyjYxyv0XDGuD5XFclfTMWqhTIJfUaM9DoVDbROElWqom6Go3hv52NMH2vSAtTKZFDzVjCo82Bi0yJqm04oSws1Ld7KTbrjuFvc9VZ5dlMa08GqvwdkuUYzmAWG\
        39jj6eTHf7C7LRjZrvlbE2IxMx3VTsDHLYCFFZLrGxblTDLqi7HRJIJgZucLfbloj4K+zbd8DnILvJKDgWAyq87Tf92GKyCLNE8mCPKKPJJkPJphtVCKh+VllbCw1Bx2Kmtlj0LM65\
        d27Ge0pqJDWU10wfxdEiPLnG6l3A9xEZC65wNy8t9KFSsXV7Qd3tScCkZ0FZd0paQfFklAFOmwjL9O3Sj6MR0tnvNE7MerehsZFtRzsDEyv8b/OU8DTim65D2ONoRb/HBYLLlSwHEi\
        ZKI2TDj8jhuiEp7pFYF+zscs7Jl+aGm6CyTG9IvAWL9IxhtQp7CR3SxX6IHq4oRUo4YC4z3cMaIk9xK+NPxEN1KSYsroPYOGFwGinb0AGWCS4sqjUuwW6rbJfKsXZTFQUwlW8uHmkl\
        hvKObHD0UCUmEKUKBFrERfDKh0rWXQZpjdrDBZalxeZ37YUau9HLtvt4E3LMM6KxUH6wT0XhNvskKGqZNW+lThgFBpu15jw7ioSQlEp2xjIFuuKs8T7Q7vjeHibc8VwXgYRcBFYsMA\
        nXK8d11+0hm29upL504GczN+p5gn04Ily0lOcaYv8PS6U5GUbtCho+0ToEZ1+GDEBlymfS040QQKoKxJi/NecIt8b40rPWCNp8/EiPrHrYJ/NtBDbONwjOCRzJRJeOUFvs2s4AOkGN\
        QBodmOqd";
    let message = cli_decrypt(sk_str, ciphertext_str);
    assert_eq!("AAAABBBBAAAABBBBAAAABBBBAAAABBBBAAAABBBBCCA=", message);
}