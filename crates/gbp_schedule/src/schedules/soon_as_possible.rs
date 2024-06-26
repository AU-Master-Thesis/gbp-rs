use crate::{GbpSchedule, GbpScheduleAtIteration, GbpScheduleIterator, GbpScheduleParams};

pub struct SoonAsPossible;

pub struct SoonAsPossibleIter {
    max: u8,
    config: GbpScheduleParams,
    internal: u8,
    external: u8,
    i: u8,
}

impl SoonAsPossibleIter {
    pub fn new(config: GbpScheduleParams) -> Self {
        let max = config.max();
        Self {
            max,
            config,
            internal: 0,
            external: 0,
            i: 0,
        }
    }
}

impl std::iter::Iterator for SoonAsPossibleIter {
    type Item = GbpScheduleAtIteration;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.max {
            let mut ts = GbpScheduleAtIteration::default();
            if self.internal < self.config.internal {
                ts.internal = true;
                self.internal += 1;
            }

            if self.external < self.config.external {
                ts.external = true;
                self.external += 1;
            }

            self.i += 1;

            Some(ts)
        } else {
            None
        }
    }
}

impl GbpScheduleIterator for SoonAsPossibleIter {}

impl GbpSchedule for SoonAsPossible {
    fn schedule(config: GbpScheduleParams) -> impl GbpScheduleIterator {
        SoonAsPossibleIter::new(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn ts(internal: bool, external: bool) -> GbpScheduleAtIteration {
        GbpScheduleAtIteration { internal, external }
    }

    #[test]
    fn internal_greater_than_external() {
        let config = GbpScheduleParams {
            internal: 10,
            external: 5,
        };
        let mut schedule = SoonAsPossible::schedule(config);
        assert_eq!(schedule.next(), Some(ts(true, true)));
        assert_eq!(schedule.next(), Some(ts(true, true)));
        assert_eq!(schedule.next(), Some(ts(true, true)));
        assert_eq!(schedule.next(), Some(ts(true, true)));
        assert_eq!(schedule.next(), Some(ts(true, true)));

        assert_eq!(schedule.next(), Some(ts(true, false)));
        assert_eq!(schedule.next(), Some(ts(true, false)));
        assert_eq!(schedule.next(), Some(ts(true, false)));
        assert_eq!(schedule.next(), Some(ts(true, false)));
        assert_eq!(schedule.next(), Some(ts(true, false)));

        assert_eq!(schedule.next(), None);
    }

    #[test]
    fn internal_less_than_external() {
        let config = GbpScheduleParams {
            internal: 3,
            external: 6,
        };
        let mut schedule = SoonAsPossible::schedule(config);
        assert_eq!(schedule.next(), Some(ts(true, true)));
        assert_eq!(schedule.next(), Some(ts(true, true)));
        assert_eq!(schedule.next(), Some(ts(true, true)));

        assert_eq!(schedule.next(), Some(ts(false, true)));
        assert_eq!(schedule.next(), Some(ts(false, true)));
        assert_eq!(schedule.next(), Some(ts(false, true)));

        assert_eq!(schedule.next(), None);
    }

    #[test]
    fn internal_external_even() {
        let config = GbpScheduleParams {
            internal: 3,
            external: 3,
        };
        let mut schedule = SoonAsPossible::schedule(config);
        assert_eq!(schedule.next(), Some(ts(true, true)));
        assert_eq!(schedule.next(), Some(ts(true, true)));
        assert_eq!(schedule.next(), Some(ts(true, true)));
        assert_eq!(schedule.next(), None);
    }

    #[test]
    fn both_zero() {
        let config = GbpScheduleParams {
            internal: 0,
            external: 0,
        };
        let mut schedule = SoonAsPossible::schedule(config);
        assert_eq!(schedule.next(), None);
    }

    #[test]
    fn internal_zero_external_not() {
        let config = GbpScheduleParams {
            internal: 0,
            external: 2,
        };
        let mut schedule = SoonAsPossible::schedule(config);
        assert_eq!(schedule.next(), Some(ts(false, true)));
        assert_eq!(schedule.next(), Some(ts(false, true)));
        assert_eq!(schedule.next(), None);
    }

    #[test]
    fn external_zero_internal_not() {
        let config = GbpScheduleParams {
            internal: 2,
            external: 0,
        };
        let mut schedule = SoonAsPossible::schedule(config);
        assert_eq!(schedule.next(), Some(ts(true, false)));
        assert_eq!(schedule.next(), Some(ts(true, false)));
        assert_eq!(schedule.next(), None);
    }
}
