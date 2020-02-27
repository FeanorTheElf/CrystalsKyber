use std::mem::MaybeUninit;

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

pub fn create_array_it<I, const N: usize>(it: &mut I) -> [I::Item; N]
    where I: Iterator
{
    unsafe {
        let mut result: MaybeUninit<[I::Item; N]> = MaybeUninit::uninit();
        let result_ptr = (*result.as_mut_ptr()).as_mut_ptr();
        for i in 0..N {
            std::ptr::write(result_ptr.offset(i as isize), it.next().unwrap());
        }
        return result.assume_init();
    }
}

pub fn create_array<T, F, const N: usize>(f: F) -> [T; N]
    where F: FnMut(usize) -> T
{
    return create_array_it(&mut (0..N).map(f));
}

pub struct CartesianIterator<I, J>
    where I: Iterator, I::Item: Clone, J: Iterator + Clone
{
    fst_iter: I,
    snd_iter: J,
    current_item: Option<I::Item>,
    current_iter: J
}

impl<I, J> Iterator for CartesianIterator<I, J>
    where I: Iterator, I::Item: Clone, J: Iterator + Clone
{
    type Item = (I::Item, J::Item);

    fn next(&mut self) -> Option<Self::Item>
    {
        while let Some(ref item) = self.current_item {
            if let Some(value) = self.current_iter.next() {
                return Some((item.clone(), value));
            } else {
                self.current_item = self.fst_iter.next();
                self.current_iter = self.snd_iter.clone();
            }
        }
        return None;
    }
}

pub fn cartesian<I, J>(mut fst: I, snd: J) -> CartesianIterator<I, J>
    where I: Iterator, I::Item: Clone, J: Iterator + Clone
{
    let item = fst.next();
    return CartesianIterator {
        fst_iter: fst,
        snd_iter: snd.clone(),
        current_item: item,
        current_iter: snd
    };
}

#[test]
fn test_shift_left() {
    assert_eq!([4, 8, 16, 32, 64, 128, 1, 2], shift_left(2, [1, 2, 4, 8, 16, 32, 64, 128]));
}
