import numpy as np
import torch

from tensorz.tensorz import TensorZ


NUM_FUZZY = 1000
np.random.seed(1337)


def test_add_forward():
    a = np.array([2.0, 4.0, -1.0, 0])
    b = np.array([-3.0, 2.0, 0.0, -4.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a + z_b

    t_a = torch.tensor(a)
    t_b = torch.tensor(b)
    t_c = t_a + t_b

    assert np.allclose(z_c.data, t_c.numpy())


def test_add_backward():
    a = np.array([2.0])
    b = np.array([-3.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a + z_b
    z_c.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = torch.tensor(b, requires_grad=True)
    t_c = t_a + t_b
    t_c.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())
    assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_add_fuzzy():
    for _ in range(NUM_FUZZY):
        a = np.random.randn(1)
        b = np.random.randn(1)

        z_a = TensorZ(a)
        z_b = TensorZ(b)
        z_c = z_a + z_b
        z_c.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = t_a + t_b
        t_c.backward()

        assert np.allclose(z_c.data, t_c.detach().numpy())
        assert np.allclose(z_a.grad, t_b.grad.numpy())
        assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_mul_forward():
    a = np.array([2.0, 4.0, -1.0, 0])
    b = np.array([-3.0, 2.0, 0.0, -4.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a * z_b

    t_a = torch.tensor(a)
    t_b = torch.tensor(b)
    t_c = t_a * t_b

    assert np.allclose(z_c.data, t_c.numpy())


def test_mul_backward():
    a = np.array([2.0])
    b = np.array([-3.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a * z_b
    z_c.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = torch.tensor(b, requires_grad=True)
    t_c = t_a * t_b
    t_c.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())
    assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_mul_fuzzy():
    for _ in range(NUM_FUZZY):
        a = np.random.randn(1)
        b = np.random.randn(1)

        z_a = TensorZ(a)
        z_b = TensorZ(b)
        z_c = z_a * z_b
        z_c.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = t_a * t_b
        t_c.backward()

        assert np.allclose(z_c.data, t_c.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())
        assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_pow_forward():
    b = np.array([0.0, 1.0, 3.0, 5.0])
    p = np.array([4.0])

    z_b = TensorZ(b)
    z_p = TensorZ(p)
    z_y = z_b**z_p

    t_b = torch.tensor(b)
    t_p = torch.tensor(p)
    t_y = t_b**t_p

    assert np.allclose(z_y.data, t_y.numpy())


def test_pow_backward():
    b = np.array([3.0])
    p = np.array([4.0])

    z_b = TensorZ(b)
    z_p = TensorZ(p)
    z_y = z_b**z_p
    z_y.backward()

    t_b = torch.tensor(b, requires_grad=True)
    t_p = torch.tensor(p, requires_grad=True)
    t_y = t_b**t_p
    t_y.backward()

    assert np.allclose(z_b.grad, t_b.grad.numpy())
    assert np.allclose(z_p.grad, t_p.grad.numpy())


def test_pow_fuzzy():
    for _ in range(NUM_FUZZY):
        b = np.random.uniform(low=0.0, high=10, size=1)
        p = np.random.uniform(low=-5.0, high=5.0, size=1)

        z_b = TensorZ(b)
        z_p = TensorZ(p)
        z_y = z_b**z_p
        z_y.backward()

        t_b = torch.tensor(b, requires_grad=True)
        t_p = torch.tensor(p, requires_grad=True)
        t_y = t_b**t_p
        t_y.backward()

        assert np.allclose(z_y.data, t_y.detach().numpy())
        assert np.allclose(z_b.grad, t_b.grad.numpy())
        assert np.allclose(z_p.grad, t_p.grad.numpy())


def test_sub_forward():
    a = np.array([2.0, 4.0, -1.0, 0])
    b = np.array([-3.0, 2.0, 0.0, -4.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a - z_b

    t_a = torch.tensor(a)
    t_b = torch.tensor(b)
    t_c = t_a - t_b

    assert np.allclose(z_c.data, t_c.numpy())


def test_sub_backward():
    a = np.array([2.0])
    b = np.array([-3.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a - z_b
    z_c.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = torch.tensor(b, requires_grad=True)
    t_c = t_a - t_b
    t_c.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())
    assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_sub_fuzzy():
    for _ in range(NUM_FUZZY):
        a = np.random.uniform(low=-100, high=100, size=1)
        b = np.random.uniform(low=-100, high=100, size=1)

        z_a = TensorZ(a)
        z_b = TensorZ(b)
        z_c = z_a - z_b
        z_c.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = t_a - t_b
        t_c.backward()

        assert np.allclose(z_c.data, t_c.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())
        assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_truediv_forward():
    a = np.array([2.0, 4.0, -1.0, 0])
    b = np.array([-3.0, 2.0, 5.0, -4.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a / z_b

    t_a = torch.tensor(a)
    t_b = torch.tensor(b)
    t_c = t_a / t_b

    assert np.allclose(z_c.data, t_c.numpy())


def test_truediv_backward():
    a = np.array([2.0])
    b = np.array([-3.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a / z_b
    z_c.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = torch.tensor(b, requires_grad=True)
    t_c = t_a / t_b
    t_c.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())
    assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_truediv_fuzzy():
    for _ in range(NUM_FUZZY):
        a = np.random.uniform(low=-100, high=100, size=1)
        b = np.random.uniform(low=1, high=100, size=1)

        z_a = TensorZ(a)
        z_b = TensorZ(b)
        z_c = z_a / z_b
        z_c.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = t_a / t_b
        t_c.backward()

        assert np.allclose(z_c.data, t_c.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())
        assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_chain_ops():
    a = np.array([3.0])
    b = np.array([2.0])
    c = np.array([-5.0])
    d = np.array([4.0])
    e = np.array([1.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = TensorZ(c)
    z_d = TensorZ(d)
    z_e = TensorZ(e)
    z_out = (-(z_a ** (z_b + z_d)) / z_c) - z_e
    z_out.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = torch.tensor(b, requires_grad=True)
    t_c = torch.tensor(c, requires_grad=True)
    t_d = torch.tensor(d, requires_grad=True)
    t_e = torch.tensor(e, requires_grad=True)
    t_out = (-(t_a ** (t_b + t_d)) / t_c) - t_e
    t_out.backward()

    assert np.allclose(z_out.data, t_out.detach().numpy())
    assert np.allclose(z_a.grad, t_a.grad.numpy())
    assert np.allclose(z_b.grad, t_b.grad.numpy())
    assert np.allclose(z_c.grad, t_c.grad.numpy())
    assert np.allclose(z_d.grad, t_d.grad.numpy())
    assert np.allclose(z_e.grad, t_e.grad.numpy())


def test_chain_ops_fuzzy():
    for _ in range(NUM_FUZZY):
        a = np.random.uniform(low=0, high=10, size=1)
        b = np.random.uniform(low=-5, high=5, size=1)
        c = np.random.uniform(low=1, high=20, size=1)
        d = np.random.uniform(low=-5, high=5, size=1)
        e = np.random.uniform(low=-100, high=100, size=1)

        z_a = TensorZ(a)
        z_b = TensorZ(b)
        z_c = TensorZ(c)
        z_d = TensorZ(d)
        z_e = TensorZ(e)
        z_out = (-(z_a ** (z_b + z_d)) / z_c) - z_e
        z_out.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = torch.tensor(c, requires_grad=True)
        t_d = torch.tensor(d, requires_grad=True)
        t_e = torch.tensor(e, requires_grad=True)
        t_out = (-(t_a ** (t_b + t_d)) / t_c) - t_e
        t_out.backward()

        assert np.allclose(z_out.data, t_out.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())
        assert np.allclose(z_b.grad, t_b.grad.numpy())
        assert np.allclose(z_c.grad, t_c.grad.numpy())
        assert np.allclose(z_d.grad, t_d.grad.numpy())
        assert np.allclose(z_e.grad, t_e.grad.numpy())


def test_ln_forward():
    a = np.array([0.5, 1.0, 4.0, 9.0])

    z_a = TensorZ(a)
    z_ln = z_a.log()

    t_a = torch.tensor(a)
    t_ln = torch.log(t_a)

    assert np.allclose(z_ln.data, t_ln.numpy())


def test_ln_backward():
    a = np.array([4.0])

    z_a = TensorZ(a)
    z_ln = z_a.log()
    z_ln.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_ln = torch.log(t_a)
    t_ln.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_ln_fuzzy():
    a = np.random.uniform(low=0.5, high=100, size=1)

    z_a = TensorZ(a)
    z_ln = z_a.log()
    z_ln.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_ln = torch.log(t_a)
    t_ln.backward()

    assert np.allclose(z_ln.data, t_ln.detach().numpy())
    assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_matmul_simple_forward():
    a = np.arange(12).reshape((3, 4))
    b = np.arange(20).reshape((4, 5))

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a @ z_b

    t_a = torch.tensor(a)
    t_b = torch.tensor(b)
    t_c = t_a @ t_b

    assert np.allclose(z_c.data, t_c.numpy())


def test_matmul_simple_backward():
    a = np.arange(12).reshape((3, 4)).astype(float)
    b = np.arange(20).reshape((4, 5)).astype(float)

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a @ z_b
    z_c.sum().backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = torch.tensor(b, requires_grad=True)
    t_c = t_a @ t_b
    t_c.sum().backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())
    assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_matmul_simple_fuzzy():
    for _ in range(NUM_FUZZY):
        a = np.random.uniform(low=-100, high=100, size=(3, 4)).astype(float)
        b = np.random.uniform(low=-100, high=100, size=(4, 5)).astype(float)

        z_a = TensorZ(a)
        z_b = TensorZ(b)
        z_c = z_a @ z_b
        z_c.sum().backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = t_a @ t_b
        t_c.sum().backward()

        assert np.allclose(z_c.data, t_c.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())
        assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_matmul_batch_forward():
    a = np.arange(36).reshape((2, 3, 6)).astype(float)
    b = np.arange(30).reshape((2, 5, 3)).astype(float)

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a @ z_b

    t_a = torch.tensor(a)
    t_b = torch.tensor(b)
    t_c = t_a @ t_b

    assert np.allclose(z_c.data, t_c.numpy())


def test_matmul_batch_backward():
    a = np.arange(36).reshape((2, 3, 6)).astype(float)
    b = np.arange(30).reshape((2, 5, 3)).astype(float)

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = z_a @ z_b
    z_c.sum().backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = torch.tensor(b, requires_grad=True)
    t_c = t_a @ t_b
    t_c.sum().backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())
    assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_matmul_batch_fuzzy():
    for _ in range(NUM_FUZZY):
        common_dim = np.random.randint(low=1, high=10)
        batch_dim = np.random.randint(low=1, high=10)
        a = np.random.uniform(
            low=-100, high=100, size=(batch_dim, common_dim, 6)
        ).astype(float)
        b = np.random.uniform(
            low=-100, high=100, size=(batch_dim, 5, common_dim)
        ).astype(float)

        z_a = TensorZ(a)
        z_b = TensorZ(b)
        z_c = z_a @ z_b
        z_c.sum().backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = t_a @ t_b
        t_c.sum().backward()

        assert np.allclose(z_c.data, t_c.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())
        assert np.allclose(z_b.grad, t_b.grad.numpy())


def test_where_forward():
    a = np.array([True, False, True, False])
    b = np.array([2.0, 4.0, -1.0, 0])
    c = np.array([-3.0, 2.0, 0.0, -4.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = TensorZ(c)
    z_d = TensorZ.where(z_a, z_b, z_c)

    t_a = torch.tensor(a)
    t_b = torch.tensor(b)
    t_c = torch.tensor(c)
    t_d = torch.where(t_a, t_b, t_c)

    assert np.allclose(z_d.data, t_d.numpy())


def test_where_backward():
    a = np.array([True, False, True, False])
    b = np.array([2.0])
    c = np.array([-3.0])

    z_a = TensorZ(a)
    z_b = TensorZ(b)
    z_c = TensorZ(c)
    z_d = TensorZ.where(z_a, z_b, z_c)
    z_d.sum().backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = torch.tensor(b, requires_grad=True)
    t_c = torch.tensor(c, requires_grad=True)
    t_d = torch.where(t_a, t_b, t_c)
    t_d.sum().backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())
    assert np.allclose(z_b.grad, t_b.grad.numpy())
    assert np.allclose(z_c.grad, t_c.grad.numpy())


def test_where_fuzzy():
    for _ in range(NUM_FUZZY):
        a = np.random.choice([True, False], size=1)
        b = np.random.uniform(low=-100, high=100, size=1)
        c = np.random.uniform(low=-100, high=100, size=1)

        z_a = TensorZ(a)
        z_b = TensorZ(b)
        z_c = TensorZ(c)
        z_d = TensorZ.where(z_a, z_b, z_c)
        z_d.sum().backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = torch.tensor(b, requires_grad=True)
        t_c = torch.tensor(c, requires_grad=True)
        t_d = torch.where(t_a, t_b, t_c)
        t_d.sum().backward()

        assert np.allclose(z_d.data, t_d.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())
        assert np.allclose(z_b.grad, t_b.grad.numpy())
        assert np.allclose(z_c.grad, t_c.grad.numpy())


def test_mean_forward():
    a = np.array([2.0, 4.0, -1.0, 0])

    z_a = TensorZ(a)
    z_b = z_a.mean()

    t_a = torch.tensor(a)
    t_b = t_a.mean()

    assert np.allclose(z_b.data, t_b.numpy())


def test_mean_backward():
    a = np.array([2.0, 4.0, -1.0, 0])

    z_a = TensorZ(a)
    z_b = z_a.mean()
    z_b.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = t_a.mean()
    t_b.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_mean_fuzzy():
    for _ in range(NUM_FUZZY):
        size = np.random.randint(low=1, high=100)
        a = np.random.uniform(low=-100, high=100, size=size)

        z_a = TensorZ(a)
        z_b = z_a.mean()
        z_b.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = t_a.mean()
        t_b.backward()

        assert np.allclose(z_b.data, t_b.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_mean_axis_forward():
    a = np.array([[2.0, 4.0, -1.0, 0], [1.0, 2.0, 3.0, 4.0]])

    z_a = TensorZ(a)
    z_b = z_a.mean(axis=1)

    t_a = torch.tensor(a)
    t_b = t_a.mean(axis=1)

    assert np.allclose(z_b.data, t_b.numpy())


def test_mean_axis_backward():
    a = np.array([[2.0, 4.0, -1.0, 0], [1.0, 2.0, 3.0, 4.0]])

    z_a = TensorZ(a)
    z_b = z_a.mean(axis=1)
    z_b.sum().backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = t_a.mean(axis=1)
    t_b.sum().backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_mean_axis_fuzzy():
    for _ in range(NUM_FUZZY):
        size = np.random.randint(low=1, high=100)
        a = np.random.uniform(low=-100, high=100, size=(size, size))

        z_a = TensorZ(a)
        z_b = z_a.mean(axis=1)
        z_b.sum().backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = t_a.mean(axis=1)
        t_b.sum().backward()

        assert np.allclose(z_b.data, t_b.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_sum_forward():
    a = np.array([2.0, 4.0, -1.0, 0])

    z_a = TensorZ(a)
    z_b = z_a.sum()

    t_a = torch.tensor(a)
    t_b = t_a.sum()

    assert np.allclose(z_b.data, t_b.numpy())


def test_sum_backward():
    a = np.array([2.0, 4.0, -1.0, 0])

    z_a = TensorZ(a)
    z_b = z_a.sum()
    z_b.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = t_a.sum()
    t_b.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_sum_fuzzy():
    for _ in range(NUM_FUZZY):
        size = np.random.randint(low=1, high=100)
        a = np.random.uniform(low=-100, high=100, size=size)

        z_a = TensorZ(a)
        z_b = z_a.sum()
        z_b.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = t_a.sum()
        t_b.backward()

        assert np.allclose(z_b.data, t_b.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_sum_axis_forward():
    a = np.array([[2.0, 4.0, -1.0, 0], [1.0, 2.0, 3.0, 4.0]])

    z_a = TensorZ(a)
    z_b = z_a.sum(axis=1)

    t_a = torch.tensor(a)
    t_b = t_a.sum(axis=1)

    assert np.allclose(z_b.data, t_b.numpy())


def test_sum_axis_backward():
    a = np.array([[2.0, 4.0, -1.0, 0], [1.0, 2.0, 3.0, 4.0]])

    z_a = TensorZ(a)
    z_b = z_a.sum(axis=1).mean()
    z_b.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = t_a.sum(axis=1).mean()
    t_b.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_sum_axis_fuzzy():
    for _ in range(NUM_FUZZY):
        size = np.random.randint(low=1, high=100)
        a = np.random.uniform(low=-100, high=100, size=(size, size))

        z_a = TensorZ(a)
        z_b = z_a.sum(axis=1).mean()
        z_b.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = t_a.sum(axis=1).mean()
        t_b.backward()

        assert np.allclose(z_b.data, t_b.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_var_forward():
    a = np.array([2.0, 4.0, -1.0, 0])

    z_a = TensorZ(a)
    z_b = z_a.var()

    t_a = torch.tensor(a)
    t_b = t_a.var()

    assert np.allclose(z_b.data, t_b.numpy())


def test_var_backward():
    a = np.array([2.0, 4.0, -1.0, 0])

    z_a = TensorZ(a)
    z_b = z_a.var()
    z_b.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = t_a.var()
    t_b.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_var_fuzzy():
    for _ in range(NUM_FUZZY):
        size = np.random.randint(low=1, high=100)
        a = np.random.uniform(low=-100, high=100, size=size)

        z_a = TensorZ(a)
        z_b = z_a.var()
        z_b.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = t_a.var()
        t_b.backward()

        assert np.allclose(z_b.data, t_b.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_std_forward():
    a = np.array([2.0, 4.0, -1.0, 0])

    z_a = TensorZ(a)
    z_b = z_a.std()

    t_a = torch.tensor(a)
    t_b = t_a.std()

    assert np.allclose(z_b.data, t_b.numpy())


def test_std_backward():
    a = np.array([2.0, 4.0, -1.0, 0])

    z_a = TensorZ(a)
    z_b = z_a.std()
    z_b.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_b = t_a.std()
    t_b.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_std_fuzzy():
    for _ in range(NUM_FUZZY):
        size = np.random.randint(low=1, high=100)
        a = np.random.uniform(low=-100, high=100, size=size)

        z_a = TensorZ(a)
        z_b = z_a.std()
        z_b.backward()

        t_a = torch.tensor(a, requires_grad=True)
        t_b = t_a.std()
        t_b.backward()

        assert np.allclose(z_b.data, t_b.detach().numpy())
        assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_std_eps():
    a = np.array([2.0, 4.0, -1.0, 0])

    z_a = TensorZ(a)
    z_b = z_a.std(eps=1e-5)

    t_a = torch.tensor(a)
    t_b = t_a.std(unbiased=False)

    assert np.allclose(z_b.data, t_b.numpy())


# TODO: test cat
