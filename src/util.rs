use std::mem::MaybeUninit;

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

pub fn try_create_array_it<I, T, E, const N: usize>(it: &mut I) -> Result<[T; N], E>
    where I: Iterator<Item = Result<T, E>>
{
    unsafe {
        let mut result: MaybeUninit<[T; N]> = MaybeUninit::uninit();
        let result_ptr = (*result.as_mut_ptr()).as_mut_ptr();
        for i in 0..N {
            std::ptr::write(result_ptr.offset(i as isize), it.next().unwrap()?);
        }
        return Ok(result.assume_init());
    }
}

pub fn try_create_array<T, E, F, const N: usize>(f: F) -> Result<[T; N], E>
    where F: FnMut(usize) -> Result<T, E>
{
    return try_create_array_it(&mut (0..N).map(f));
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

#[allow(unused)]
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
