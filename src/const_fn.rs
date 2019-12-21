
pub const fn shift_left(amount: usize, values: [i32; 8]) -> [i32; 8]
{
    [
        values[amount % 8], 
        values[(amount + 1) % 8], 
        values[(amount + 2) % 8], 
        values[(amount + 3) % 8], 
        values[(amount + 4) % 8],
        values[(amount + 5) % 8], 
        values[(amount + 6) % 8], 
        values[(amount + 7) % 8]
    ]
}

#[test]
fn test_shift_left() {
    assert_eq!([4, 8, 16, 32, 64, 128, 1, 2], shift_left(2, [1, 2, 4, 8, 16, 32, 64, 128]));
}