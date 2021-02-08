class LinearScheduler:
    """
    class represent Linear Scheduler y = a * x
    """
    def __init__(self, start, end=None, coefficient=None):
        self.start = start
        self.end = end
        self.coefficient = coefficient
        self.current = start

    def step(self):
        assert self.coefficient is not None, "coefficient is None"

        if self.end is None:
            self.current += self.coefficient
        else:
            if abs(self.current - self.end) > 1e-8:
                self.current += self.coefficient
            else:
                self.current = self.end

        return self.current

    def calc_coefficient(self, param_val, epoch, iter_on_epoch):
        self.coefficient = param_val / (epoch * iter_on_epoch)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"

    ls = LinearScheduler(1, 0.5)
    ls.calc_coefficient(-0.5, 50, 300)
    for i in range(50*300):
        print(f"iter {i} - {ls.step()}")
    print(f"current: {ls.current}")
    print(ls.coefficient)
