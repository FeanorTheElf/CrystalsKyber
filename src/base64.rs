use super::util;

struct Buffer
{
    length: usize,
    buffer: u32
}

impl Buffer
{
    fn read_bits(&mut self, bits: usize) -> u16
    {
        assert!(bits <= 16);
        assert!(bits <= self.length);
        let result = self.buffer >> (self.length - bits);
        self.length -= bits;
        self.buffer = self.buffer & ((1 << self.length) - 1);
        return result as u16;
    }

    fn write_bits(&mut self, bits: usize, data: u16)
    {
        assert!(bits <= 16);
        assert!(bits + self.length <= 32);
        self.buffer = (self.buffer << bits) | data as u32;
        self.length += bits;
    }
}

pub struct Encoder
{
    current: String,
    current_buffer: Buffer
}

const BITS_PER_SYMBOL: usize = 6;

impl Encoder
{
    pub fn new() -> Encoder
    {
        Encoder {
            current: "".to_owned(),
            current_buffer: Buffer {
                length: 0,
                buffer: 0
            }
        }
    }

    fn append_symbol(&mut self, symbol: u8)
    {
        if symbol < 26 {
            self.current.push(char::from(65 + symbol));
        } else if symbol < 52 {
            self.current.push(char::from(97 + (symbol - 26)));
        } else if symbol < 62 {
            self.current.push(char::from(48 + (symbol - 52)));
        } else if symbol == 62 {
            self.current.push(char::from(43));
        } else {
            self.current.push(char::from(47));
        }
        
    }

    pub fn encode_bits(&mut self, bits: u16, bit_count: usize)
    {
        assert!(bit_count <= 16);
        self.current_buffer.write_bits(bit_count, bits);
        while self.current_buffer.length >= BITS_PER_SYMBOL {
            let new_symbol = self.current_buffer.read_bits(BITS_PER_SYMBOL) as u8;
            self.append_symbol(new_symbol);
        }
    }

    pub fn encode(&mut self, byte: u8)
    {
        self.encode_bits(byte as u16, 8);
    }

    pub fn get(mut self) -> String
    {
        if self.current_buffer.length != 0 {
            let buffer_len = self.current_buffer.length;
            let new_symbol = (self.current_buffer.read_bits(buffer_len) as u8) << (BITS_PER_SYMBOL - buffer_len);
            self.append_symbol(new_symbol);
        }
        while self.current.len() % 4 != 0 {
            self.current.push('=');
        }
        return self.current;
    }

    pub fn encode_bytes(&mut self, bytes: &[u8])
    {
        for byte in bytes {
            self.encode(*byte);
        }
    }
}

pub struct Decoder
{
    current: String,
    current_buffer: Buffer
}

impl Decoder
{
    pub fn new(data: &str) -> Decoder
    {
        Decoder {
            current: data.chars().rev().collect(),
            current_buffer: Buffer {
                length: 0,
                buffer: 0
            }
        }
    }

    fn read_symbol(&mut self) -> u8 
    {
        let c = self.current.pop().unwrap();
        let offset: i16 = match c {
            'a'..='z' => -71,
            'A'..='Z' => -65,
            '0'..='9' => 4,
            '+' => 19,
            '/' => 16,
            _ => panic!("Unknown character {}", c)
        };
        let mut result: [u8; 1] = [0; 1];
        c.encode_utf8(&mut result);
        return (result[0] as i16 + offset) as u8;
    }

    pub fn read_bits(&mut self, bit_count: usize) -> u16
    {
        assert!(bit_count <= 16);
        while self.current_buffer.length < bit_count {
            let new_bits = self.read_symbol() as u16;
            self.current_buffer.write_bits(BITS_PER_SYMBOL, new_bits);
        }
        return self.current_buffer.read_bits(bit_count);
    }

    pub fn read(&mut self) -> u8
    {
        self.read_bits(8) as u8
    }

    pub fn read_bytes<const N: usize>(&mut self) -> [u8; N]
    {
        util::create_array(|_i| self.read())
    }
}

impl Iterator for Decoder
{
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        let last_char = self.current.pop();
        if let Some(c) = last_char {
            self.current.push(c);
        }
        if self.current_buffer.length >= 8 || last_char.map(|c| c != '=').unwrap_or(false) {
            Some(self.read())
        } else {
            None
        }
    }
}

#[test]
fn test_base64_encode_decode() {
    let mut enc = Encoder::new();
    enc.encode(65);
    enc.encode(97);
    enc.encode(3);
    enc.encode(255);
    let code = enc.get();
    let mut decoder = Decoder::new(code.as_str());
    assert_eq!(65, decoder.read());
    assert_eq!(97, decoder.read());
    assert_eq!(3, decoder.read());
    assert_eq!(255, decoder.read());
}

