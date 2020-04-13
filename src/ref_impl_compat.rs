use super::encoding::*;
use super::kyber::*;
use super::rqvec::*;
use super::ring::*;
use super::zq::*;
use super::util;

// in the reference implementation, some strange order of the elements in chinese remainder representation
// is used (i.e. a strange order of the primitive 512-th roots of unities); Use this permutation to convert
// the secret key to the value expected by their implementation.
// our_vector[CONVERT_PERMUTATION_INDICES[i]] = their_vector[i]
const CONVERT_PERMUTATION_INDICES: [usize; 256] = [140, 12, 204, 76, 172, 44, 236, 108, 28, 156, 92, 220, 60, 188, 
    124, 252, 84, 212, 148, 20, 116, 244, 180, 52, 228, 100, 36, 164, 4, 132, 68, 196, 240, 112, 48, 176, 16, 144, 
    80, 208, 128, 0, 192, 64, 160, 32, 224, 96, 184, 56, 248, 120, 216, 88, 24, 152, 72, 200, 136, 8, 104, 232, 168, 
    40, 190, 62, 254, 126, 222, 94, 30, 158, 78, 206, 142, 14, 110, 238, 174, 46, 134, 6, 198, 70, 166, 38, 230, 102, 
    22, 150, 86, 214, 54, 182, 118, 246, 34, 162, 98, 226, 66, 194, 130, 2, 178, 50, 242, 114, 210, 82, 18, 146, 234, 
    106, 42, 170, 10, 138, 74, 202, 122, 250, 186, 58, 154, 26, 218, 90, 165, 37, 229, 101, 197, 69, 5, 133, 53, 181,
    117, 245, 85, 213, 149, 21, 109, 237, 173, 45, 141, 13, 205, 77, 253, 125, 61, 189, 29, 157, 93, 221, 9, 137, 73, 
    201, 41, 169, 105, 233, 153, 25, 217, 89, 185, 57, 249, 121, 209, 81, 17, 145, 241, 113, 49, 177, 97, 225, 161, 
    33, 129, 1, 193, 65, 215, 87, 23, 151, 247, 119, 55, 183, 103, 231, 167, 39, 135, 7, 199, 71, 159, 31, 223, 95, 
    191, 63, 255, 127, 47, 175, 111, 239, 79, 207, 143, 15, 59, 187, 123, 251, 91, 219, 155, 27, 203, 75, 11, 139, 235, 
    107, 43, 171, 3, 131, 67, 195, 35, 163, 99, 227, 147, 19, 211, 83, 179, 51, 243, 115];

// See CONVERT_PERMUTATION_INDICES
const CONVERT_INV_PERMUTATION_INDICES: [usize; 256] = [41, 189, 103, 240, 28, 134, 81, 205, 59, 160, 116, 234, 1, 149, 
    75, 223, 36, 178, 110, 249, 19, 143, 88, 194, 54, 169, 125, 231, 8, 156, 70, 209, 45, 187, 96, 244, 26, 129, 85, 203, 
    63, 164, 114, 238, 5, 147, 79, 216, 34, 182, 105, 253, 23, 136, 92, 198, 49, 173, 123, 224, 12, 154, 65, 213, 43, 191, 
    100, 242, 30, 133, 83, 207, 56, 162, 118, 233, 3, 151, 72, 220, 38, 177, 109, 251, 16, 140, 90, 193, 53, 171, 127, 228, 
    10, 158, 69, 211, 47, 184, 98, 246, 25, 131, 87, 200, 60, 166, 113, 237, 7, 144, 76, 218, 33, 181, 107, 255, 20, 138, 94, 
    197, 51, 175, 120, 226, 14, 153, 67, 215, 40, 188, 102, 241, 29, 135, 80, 204, 58, 161, 117, 235, 0, 148, 74, 222, 37, 179, 
    111, 248, 18, 142, 89, 195, 55, 168, 124, 230, 9, 157, 71, 208, 44, 186, 97, 245, 27, 128, 84, 202, 62, 165, 115, 239, 4, 
    146, 78, 217, 35, 183, 104, 252, 22, 137, 93, 199, 48, 172, 122, 225, 13, 155, 64, 212, 42, 190, 101, 243, 31, 132, 82, 206, 
    57, 163, 119, 232, 2, 150, 73, 221, 39, 176, 108, 250, 17, 141, 91, 192, 52, 170, 126, 229, 11, 159, 68, 210, 46, 185, 99, 
    247, 24, 130, 86, 201, 61, 167, 112, 236, 6, 145, 77, 219, 32, 180, 106, 254, 21, 139, 95, 196, 50, 174, 121, 227, 15, 152, 
    66, 214];

pub fn base64_encode<'a>(result: &'a mut String) -> impl Encoder + 'a
{
    let mut base64_encoder = Base64Encoder::new(move |c| result.push(c));
    ByteStreamEncoder::new(move |byte| base64_encoder.encode(byte))
}

pub fn base64_decode<'a>(data: &'a str) -> impl Decoder + 'a
{
    let mut input_iter = data.chars();
    let mut base64_decoder = Base64Decoder::new(move || input_iter.next());
    ByteStreamDecoder::new(move || base64_decoder.read())
}

pub fn read_sk_from_ref_impl(data: &str) -> SecretKey
{
    let mut secret_key_decoder = base64_decode(data);
    let secret_key_their_crt_data: [[ZqElement; N]; DIM] = util::create_array(|_|
        util::create_array(|_|
            ZqElement::from_perfect(secret_key_decoder.read_bits(13).expect("Input too short") as i16)
        )
    );
    let secret_key: [[ZqElement; N]; DIM] = util::create_array(|i|
        util::create_array(|j|
            secret_key_their_crt_data[i][CONVERT_INV_PERMUTATION_INDICES[j]]
        )
    ); 
    return RqVector::from(util::create_array(|i|
        <RqElement as RqElementCoefficientRepr>::ChineseRemainderRepr::from(secret_key[i])
    ));
}

pub fn write_sk_to_ref_impl(secret_key: &SecretKey) -> String
{
    let mut result: String = String::new();
    {
        let mut encoder = base64_encode(&mut result);
        for i in 0..DIM {
            for j in 0..N {
                encoder.encode_bits(secret_key[i].value_at_zeta(j).representative_pos() as u16, 13)
            }
        }
    }
    return result;
}

pub fn read_pk_from_ref_impl(public_key: &str) -> PublicKey
{
    let mut decoder = base64_decode(public_key);
    return (CompressedRqVector::decode(&mut decoder), decoder.read_bytes().expect("Input too short"));
}

pub fn write_pk_to_ref_impl(public_key: &PublicKey) -> String
{
    let mut result: String = String::new();
    {
        let mut encoder = base64_encode(&mut result);
        public_key.0.encode(&mut encoder);
        encoder.encode_bytes(&public_key.1);
    }
    return result;
}

pub fn read_ciphertext_from_ref_impl(ciphertext: &str) -> Ciphertext
{
    let mut decoder = base64_decode(ciphertext);
    return (CompressedRqVector::decode(&mut decoder), CompressedRq::decode(&mut decoder));
}

pub fn write_ciphertext_to_ref_impl(ciphertext: &Ciphertext) -> String
{
    let mut result: String = String::new();
    {
        let mut encoder = base64_encode(&mut result);
        ciphertext.0.encode(&mut encoder);
        ciphertext.1.encode(&mut encoder);
    }
    return result;
}

pub fn read_message_from_ref_impl(plaintext: &str) -> Plaintext
{
    let mut decoder = base64_decode(plaintext);
    return decoder.read_bytes().expect("Input too short");
}

pub fn write_message_to_ref_impl(plaintext: &Plaintext) -> String
{
    let mut result: String = String::new();
    {
        let mut encoder = base64_encode(&mut result);
        encoder.encode_bytes(plaintext);
    }
    return result;
}

#[test]
fn test_read_write_sk() {
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
    let sk = read_sk_from_ref_impl(sk_str);
    assert_eq!(sk_str, write_sk_to_ref_impl(&sk));
}

#[test]
fn test_read_write_pk() {
    let pk_str = "\
        2P6Y6UMA/yXf1QuK6fUebxNWSqp8NSioRmLhOzYEPN8AxWUsG9yX9IZJ6OYjbSo7T4o5hCO9C2YJeI7h2s/qr+KyDOXs/7xI6vY9qAKxVuStelhfhyk2UFd8+siLSfzy7p/6vjk\
        TnEIUdt4wSs1nBiOW9Y4043qHro+q/Wfz+Of+4aWjI48vb38RkEj0dx9WPP+y9WS767mdv0jDiS2tJhlPA4Ui+j0VeZHXy89U/ZxXbr9oFR7FfOZkyZYu/24aNb/0Qj4H8szLWq\
        vZN1FIdw8ZwNgPfege7drLPPurKnCD83jUXRNENyZVwh0pQULGIhIEA8mBHsQgTgK7XJNbSJ/eVlWzc17DhNOQLVo8n0n4Xt6n3MY+QWxR14a4hVibuuh7Gxl7e/0t2dT6/1bQL\
        mEeRl3MOllG16KVq0VYnXLeWjYEKCwDnSIK98A98pqAGDaEvhdploP2/E8eJ+ljdUdtrUW06p8Kwuaa48whfsuHbQx9WPB1a0fDYpb4a9TdyUf+bte6j0fofQTW+pu/CjDZuAqT\
        UsaJkCsC7k8wsbLxsqCge65fFezqVvewZwoxDp7UAaFm2Bo1YjN9yLhdFftDWcbv+crqHlTDDxKxcgFf6ED3czcF9mTv+cvlCs8VgyBNCVsoF3waPxpMxa8kBeG1Q17H8mdzI5t\
        XJ48kuDs3EU0OjUUAVaAQdAmaKs7IqBnnDgmjH5REjYpJWCK80JTD99VO5GLQdaz6IBZ413ZrEjwAThOZinAbQ3n6WayHBqWOZESpUJPfUJgs5z8rsdWflPFJZpHTx88zWC3KaN\
        89ejZnZg9tk/aE1OmPxaIuaOctvBRay6RbiIrVYy6FxjMUipI+649LQ03PZQ7WS4NKDmxRRRFaQxNGOWtrpYUn7CCrEnKZb53OraffYNTPgmuZe5OXSxi+j/M7f5o+HvcANbxZk\
        2AN1rpjbn0/sKLSPSOVU32Vax79OdZz9B0j7+ys2upMciygVNZDo9q82lDTBE/yLdlgCshbCRK6y4OqWuJj1JX0ZzOc/8m8rRHj5RFOyO7Q1bGwL51UMcpXmpegVUMxlZ37QBJQ\
        Fsy/p3HRFcgGTrSBa3p5N1qHOMOV7DC3VvX+YajKf5obMXcPDpJdhhEkY3OZe34DAn4ZCquvN2SNbkWYo6n07YA76WssJSFrDcbuqDRuU1NRxjgJUGet19+ItOOVGcfp/tWWg1M\
        3SKfp3fR3nXmJzfBVOPnQq0wazMcaoT0O5oLY/9rWl6JF+URu9HwCkC0M99qhRNFK7qGf/XV6C3+lZcvO9LdPtjAKfKh0TBRZSx6ADu3FR9hQ+VBqn5JPThyZYEMql7SJdMdROX\
        PLLVO29vwmC+eNis7VSHjzDa14TL8x1KnqL5hHAPB5Md2v3enquzF/Go/6pXvMFaWDnLo8Io54OInP3q+a+XEHSLZxWotXpefN8Vo=";
    let pk = read_pk_from_ref_impl(pk_str);
    assert_eq!(pk_str, write_pk_to_ref_impl(&pk));
}

#[test]
fn test_read_write_ciphertext() {
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
    let ciphertext = read_ciphertext_from_ref_impl(ciphertext_str);
    assert_eq!(ciphertext_str, write_ciphertext_to_ref_impl(&ciphertext));
}

#[test]
fn test_read_message_ciphertext() {
    let message_str = "AAAABBBBAAAABBBBAAAABBBBAAAABBBBAAAABBBBCCA=";
    let message = read_message_from_ref_impl(message_str);
    assert_eq!(message_str, write_message_to_ref_impl(&message));
}