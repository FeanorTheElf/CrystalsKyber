use super::util;

use std::ops::Try;

/// Queue containing up to 32 bits
struct BigEndianBitQueue
{
    length: usize,
    // aligned so that the least significant length bits are filled 
    buffer: u32
}

impl BigEndianBitQueue
{
    fn new() -> Self
    {
        BigEndianBitQueue {
            length: 0,
            buffer: 0
        }
    }

    /// Dequeue bits from the front of the queue
    fn read_bits(&mut self, bits: usize) -> u16
    {
        assert!(bits <= 16);
        assert!(bits <= self.length);
        let result = self.buffer >> (self.length - bits);
        self.length -= bits;
        self.buffer = self.buffer & ((1 << self.length) - 1);
        return result as u16;
    }

    // Enqueue bits to the end of the queue
    fn write_bits(&mut self, bits: usize, data: u16)
    {
        assert!(bits <= 16);
        assert!(bits + self.length <= 32);
        self.buffer = (self.buffer << bits) | data as u32;
        self.length += bits;
    }

    fn len(&self) -> usize
    {
        self.length
    }
}

/// Queue containing up to 32 bits
struct LittleEndianBitQueue
{
    length: usize,
    // aligned so that the least significant length bits are filled
    buffer: u32
}

impl LittleEndianBitQueue
{
    fn new() -> Self
    {
        LittleEndianBitQueue {
            length: 0,
            buffer: 0
        }
    }

    /// Dequeue bits from the front of the queue
    fn read_bits(&mut self, bits: usize) -> u16
    {
        assert!(bits <= 16);
        assert!(bits <= self.length);
        let result = self.buffer & ((1 << bits) - 1);
        self.length -= bits;
        self.buffer = self.buffer >> bits;
        return result as u16;
    }

    // Enqueue bits to the end of the queue
    fn write_bits(&mut self, bits: usize, data: u16)
    {
        assert!(bits <= 16);
        assert!(bits + self.length <= 32);
        self.buffer |= (data as u32) << self.length;
        self.length += bits;
    }

    fn len(&self) -> usize
    {
        self.length
    }
}

pub trait Encoder
{
    fn encode_bits(&mut self, bits: u16, bit_count: usize);

    fn encode(&mut self, byte: u8)
    {
        self.encode_bits(byte as u16, 8);
    }

    fn encode_bytes(&mut self, bytes: &[u8])
    {
        for byte in bytes {
            self.encode(*byte);
        }
    }
}

pub struct ByteStreamEncoder<C>
    where C: FnMut(u8)
{
    consumer: C,
    queue: LittleEndianBitQueue
}

impl<C> ByteStreamEncoder<C>
    where C: FnMut(u8)
{
    fn new(consumer: C) -> Self
    {
        ByteStreamEncoder {
            consumer: consumer,
            queue: LittleEndianBitQueue::new()
        }
    }
}

impl<C> Encoder for ByteStreamEncoder<C>
    where C: FnMut(u8)
{
    fn encode_bits(&mut self, bits: u16, bit_count: usize)
    {
        self.queue.write_bits(bit_count, bits);
        while self.queue.len() >= 8 {
            (self.consumer)(self.queue.read_bits(8) as u8);
        }
    }
}

pub struct ByteStreamDecoder<P>
    where P: FnMut() -> Option<u8>
{
    producer: P,
    queue: LittleEndianBitQueue
}

impl<P> ByteStreamDecoder<P>
    where P: FnMut() -> Option<u8>
{
    fn new(producer: P) -> Self
    {
        ByteStreamDecoder {
            producer: producer,
            queue: LittleEndianBitQueue::new()
        }
    }
}

impl<P> Decoder for ByteStreamDecoder<P>
    where P: FnMut() -> Option<u8>
{
    fn read_bits(&mut self, bit_count: usize) -> Option<u16>
    {
        while self.queue.len() < bit_count {
            self.queue.write_bits(8, (self.producer)()? as u16);
        }
        Some(self.queue.read_bits(bit_count))
    }
}

pub struct Base64Encoder<C>
    where C: FnMut(char)
{
    consumer: C,
    current_buffer: BigEndianBitQueue,
    symbol_count_mod_4: u8
}

impl<C> Base64Encoder<C>
    where C: FnMut(char)
{
    fn new(consumer: C) -> Self
    {
        Base64Encoder {
            consumer: consumer,
            current_buffer: BigEndianBitQueue::new(),
            symbol_count_mod_4: 0
        }
    }

    fn append_symbol(&mut self, symbol: u8)
    {
        self.symbol_count_mod_4 = (self.symbol_count_mod_4 + 1) & 0x3;
        if symbol < 26 {
            (self.consumer)(char::from(65 + symbol));
        } else if symbol < 52 {
            (self.consumer)(char::from(97 + (symbol - 26)));
        } else if symbol < 62 {
            (self.consumer)(char::from(48 + (symbol - 52)));
        } else if symbol == 62 {
            (self.consumer)(char::from(43));
        } else {
            (self.consumer)(char::from(47));
        }
    }
}

impl<C> std::ops::Drop for Base64Encoder<C>
    where C: FnMut(char)
{
    fn drop(&mut self)
    {
        if self.current_buffer.length != 0 {
            let buffer_len = self.current_buffer.length;
            let new_symbol = (self.current_buffer.read_bits(buffer_len) as u8) << (6 - buffer_len);
            self.append_symbol(new_symbol);
        }
        for _ in self.symbol_count_mod_4..4 {
            (self.consumer)('=');
        }
    }
}

impl<C> Encoder for Base64Encoder<C>
    where C: FnMut(char)
{
    fn encode_bits(&mut self, bits: u16, bit_count: usize)
    {
        self.current_buffer.write_bits(bit_count, bits);
        while self.current_buffer.len() >= 6 {
            let new_symbol = self.current_buffer.read_bits(6) as u8;
            self.append_symbol(new_symbol);
        }
    }
}

pub trait Decoder
{
    fn read_bits(&mut self, bit_count: usize) -> Option<u16>;

    fn read(&mut self) -> Option<u8>
    {
        self.read_bits(8).map(|x| x as u8)
    }

    fn read_bytes<const N: usize>(&mut self) -> Option<[u8; N]>
    {
        util::try_create_array(|_i| self.read().into_result()).ok()
    }
}

pub struct Base64Decoder<P>
    where P: FnMut() -> Option<char>
{
    producer: P,
    current_buffer: BigEndianBitQueue
}

impl<P> Base64Decoder<P>
    where P: FnMut() -> Option<char>
{
    pub fn new(producer: P) -> Base64Decoder<P>
    {
        Base64Decoder {
            producer: producer,
            current_buffer: BigEndianBitQueue::new(),
        }
    }

    fn read_symbol(&mut self) -> Option<u8> 
    {
        let c = (self.producer)()?;
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
        return Some((result[0] as i16 + offset) as u8);
    }
}

impl<P> Decoder for Base64Decoder<P>
    where P: FnMut() -> Option<char>
{
    fn read_bits(&mut self, bit_count: usize) -> Option<u16>
    {
        while self.current_buffer.len() < bit_count {
            let new_bits = self.read_symbol()? as u16;
            self.current_buffer.write_bits(6, new_bits);
        }
        let result = self.current_buffer.read_bits(bit_count);
        return Some(result);
    }
}

pub trait Encodable: Sized
{
    fn encode<T: Encoder>(&self, encoder: &mut T);
    fn decode<T: Decoder>(data: &mut T) -> Self;
}

pub fn base64_encode<'a>(result: &'a mut String) -> impl Encoder + 'a
{
    let mut base64_encoder = Base64Encoder::new(move |c| result.push(c));
    ByteStreamEncoder::new(move |byte| base64_encoder.encode(byte))
}

pub fn base64_decode<'a>(result: &'a str) -> impl Decoder + 'a
{
    let mut input_iter = result.chars();
    let mut base64_decoder = Base64Decoder::new(move || input_iter.next());
    ByteStreamDecoder::new(move || base64_decoder.read())
}

#[test]
fn test_base64_encode_decode() {
    let mut buffer: Vec<char> = Vec::new();
    {
        let mut encoder = Base64Encoder::new(|c| buffer.push(c));
        encoder.encode(65);
        encoder.encode(97);
        encoder.encode(3);
        encoder.encode(255);
    }
    buffer.reverse();
    let mut decoder = Base64Decoder::new(|| buffer.pop());
    assert_eq!(65, decoder.read().unwrap());
    assert_eq!(97, decoder.read().unwrap());
    assert_eq!(3, decoder.read().unwrap());
    assert_eq!(255, decoder.read().unwrap());
}

#[test]
fn test_bit_queue() {
    let mut queue = LittleEndianBitQueue::new();
    queue.write_bits(3, 0b110);
    queue.write_bits(5, 0b01110);
    assert_eq!(0b1110110, queue.read_bits(7));
    assert_eq!(0b0, queue.read_bits(1));
}

#[test]
fn test_big_endian_bit_queue() {
    let mut queue = BigEndianBitQueue::new();
    queue.write_bits(6, 63);
    queue.write_bits(6, 3 << 4);
    assert_eq!(255, queue.read_bits(8));
}