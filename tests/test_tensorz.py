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
    z_y = z_b ** z_p

    t_b = torch.tensor(b)
    t_p = torch.tensor(p)
    t_y = t_b ** t_p

    assert np.allclose(z_y.data, t_y.numpy())


def test_pow_backward():
    b = np.array([3.0])
    p = np.array([4.0])

    z_b = TensorZ(b)
    z_p = TensorZ(p)
    z_y = z_b ** z_p
    z_y.backward()

    t_b = torch.tensor(b, requires_grad=True)
    t_p = torch.tensor(p, requires_grad=True)
    t_y = t_b ** t_p
    t_y.backward()

    assert np.allclose(z_b.grad, t_b.grad.numpy())
    assert np.allclose(z_p.grad, t_p.grad.numpy())


def test_pow_fuzzy():
    for _ in range(NUM_FUZZY):
        b = np.random.uniform(low=0.0, high=10, size=1)
        p = np.random.uniform(low=-5.0, high=5.0, size=1)

        z_b = TensorZ(b)
        z_p = TensorZ(p)
        z_y = z_b ** z_p
        z_y.backward()

        t_b = torch.tensor(b, requires_grad=True)
        t_p = torch.tensor(p, requires_grad=True)
        t_y = t_b ** t_p
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
    z_ln = z_a.ln()

    t_a = torch.tensor(a)
    t_ln = torch.log(t_a)

    assert np.allclose(z_ln.data, t_ln.numpy())


def test_ln_backward():
    a = np.array([4.0])

    z_a = TensorZ(a)
    z_ln = z_a.ln()
    z_ln.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_ln = torch.log(t_a)
    t_ln.backward()

    assert np.allclose(z_a.grad, t_a.grad.numpy())


def test_ln_fuzzy():
    a = np.random.uniform(low=0.5, high=100, size=1)

    z_a = TensorZ(a)
    z_ln = z_a.ln()
    z_ln.backward()

    t_a = torch.tensor(a, requires_grad=True)
    t_ln = torch.log(t_a)
    t_ln.backward()

    assert np.allclose(z_ln.data, t_ln.detach().numpy())
    assert np.allclose(z_a.grad, t_a.grad.numpy())

