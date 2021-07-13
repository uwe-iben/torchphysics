import pytest
import torch
import numpy as np
from neural_diff_eq.utils.differentialoperators import (laplacian, 
                                                        grad, 
                                                        normal_derivative, 
                                                        div, 
                                                        jac, 
                                                        rot,
                                                        partial)

# Test laplace-operator
def function(a):
    out = 0
    for i in range(len(a)):
        out += a[i]**2
    return out


def test_laplacian_for_one_input():
    a = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output = function(a[0])
    l = laplacian(output, a)
    assert l.shape[0] == 1
    assert l.shape[1] == 1
    assert l.detach().numpy()[0] == 4


def test_laplacian_for_many_inputs():
    a = torch.tensor([[1.0, 1.0], [2.0, 3.4], [1.3, 2], [0, 0]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [4, 4, 4, 4])



def test_laplacian_in_1D():
    a = torch.tensor([[1.0], [2.0], [1.3]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [2, 2, 2])



def test_laplacian_in_3D():
    a = torch.tensor([[1.0, 3.4, 1.0], [2.0, 0, 0], [1.3, 9, 1]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [6, 6, 6])



def test_laplacian_for_complexer_function_1():
    a = torch.tensor([[1.0, 1.0, 1.0], [2.0, 1.0, 0], [0, 0, 0], [1.0, 0, 4.0]],
                     requires_grad=True)
    def function1(a):
        return a[0]**2 + a[1]**3 + 4*a[2]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]) : output[i] = function1(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[32], [8], [2], [98]])


def test_laplacian_for_complexer_function_2():
    a = torch.tensor([[1.0, 1.0], [2.0, 0], [0, 0],
                      [0, 4.0], [2, 2]], requires_grad=True)
    def function1(a):
        return a[0]**3 + torch.sin(a[1])
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.allclose(l.detach().numpy(), [[6-np.sin(1)], [12], [0],
                                            [-np.sin(4)], [12-np.sin(2)]])


def test_laplacian_for_two_inputs_one_linear():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    def function1(a, b):
        return 2*a[0]**2 + a[1]**2 + b[0]
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [6, 6])  
    l = laplacian(output, b)
    assert l.shape[0] == b.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [0, 0])  



def test_laplacian_for_two_not_linear_inputs():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 + a[1]**2 + b[0]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [4, 4])  
    l = laplacian(output, b)
    assert l.shape[0] == b.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[6], [3]]) 



def test_laplacian_multiply_varibales():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0], [2]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 * a[1]**2 * b[0]**2
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[4], [32]])  
    l = laplacian(output, b)
    assert l.shape[0] == b.shape[0]
    assert l.shape[1] == 1
    assert np.all(l.detach().numpy() == [[2], [0]])    



def test_laplacian_with_chain_rule():
    a = torch.tensor([[1.0, 1], [2.0, 1]], requires_grad=True)
    def function1(a):
        return torch.sin(2.0 * (torch.sin(a[0]))) * a[1]
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.allclose(l.detach().numpy(), [[-0.97203], [-0.22555]], atol=1e-04)  


def test_laplacian_with_tanh():
    a = torch.tensor([[1.0, 1.0, 2.0], [2.0, 0, 1.0]], requires_grad=True)
    def function1(a):
        return torch.tanh(a[0]**2 * a[1]**3 + a[2]**2)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i])
    l = laplacian(output, a)
    assert l.shape[0] == a.shape[0]
    assert l.shape[1] == 1
    assert np.allclose(l.detach().numpy(), [[-0.0087], [-1.7189]], atol=1e-04)  


# Test gradient
def test_gradient_for_one_input():
    a = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output = function(a[0])
    g = grad(output, a)
    assert g.shape[0] == 1
    assert g.shape[1] == 2
    assert np.equal(g.detach().numpy(), [2, 2]).all()


def test_gradient_many_inputs():
    a = torch.tensor([[1.0, 1.0], [2, 0], [3, 1]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    g = grad(output, a)
    assert g.shape[0] == 3
    assert g.shape[1] == 2
    assert np.equal(g.detach().numpy(), [[2, 2], [4, 0], [6, 2]]).all()   


def test_gradient_1D():
    a = torch.tensor([[1.0], [2.0], [0]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    g = grad(output, a)
    assert g.shape[0] == 3
    assert g.shape[1] == 1
    assert np.equal(g.detach().numpy(), [[2], [4], [0]]).all()


def test_gradient_3D():
    a = torch.tensor([[1.0, 5, 2], [2.0, 2.0, 2.0]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    g = grad(output, a)
    assert g.shape[0] == 2
    assert g.shape[1] == 3
    assert np.equal(g.detach().numpy(), [[2, 10, 4], [4, 4, 4]]).all()


def test_gradient_mixed_input():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0], [0.5]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 + a[1] + b[0]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    g = grad(output, a)
    assert g.shape[0] == a.shape[0]
    assert g.shape[1] == 2
    assert np.equal(g.detach().numpy(), [[2, 1], [4, 1]]).all()  
    g = grad(output, b)
    assert g.shape[0] == b.shape[0]
    assert g.shape[1] == 1
    assert np.equal(g.detach().numpy(), [[3], [3/4]]).all() 


# Test normal derivative
def test_normal_derivative_for_one_input():
    a = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output = function(a[0])
    normal = torch.tensor([[1.0, 0]])
    n = normal_derivative(output, a, normal)
    assert n.shape[0] == 1
    assert n.shape[1] == 1
    assert np.equal(n.detach().numpy(), [2]).all()


def test_normal_derivative_for_many_inputs():
    a = torch.tensor([[1.0, 1.0], [0, 1], [2, 3]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    normals = torch.tensor([[1.0, 0], [1.0, 0], [np.cos(np.pi/4), np.sin(np.pi/4)]])
    n = normal_derivative(output, a, normals)
    assert n.shape[0] == 3
    assert n.shape[1] == 1
    assert np.allclose(n.detach().numpy(), [[2], [0],
                                            [4*np.cos(np.pi/4)+6*np.sin(np.pi/4)]])


def test_normal_derivative_3D():
    a = torch.tensor([[1.0, 1.0, 1.0], [0, 1, 2]], requires_grad=True)
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function(a[i])
    normals = torch.tensor([[1.0, 0, 0], [1.0, 0, 1.0]])
    n = normal_derivative(output, a, normals)
    assert n.shape[0] == 2
    assert n.shape[1] == 1
    assert np.allclose(n.detach().numpy(), [[2], [4]])


def test_normal_derivative_complexer_function():
    a = torch.tensor([[1.0, 1.0], [2.0, 0]], requires_grad=True)
    b = torch.tensor([[1.0, 2.0], [3.0, 3.0]], requires_grad=True)
    def function1(a, b):
        return a[0]**2 + torch.sin(a[1]) + b[0]**3
    output = torch.zeros(a.shape[0])
    for i in range(a.shape[0]):
        output[i] = function1(a[i], b[i])
    normals = torch.tensor([[1.0, 0], [1.0/np.sqrt(2), 1.0/np.sqrt(2)]])
    n = normal_derivative(output, a, normals)
    assert n.shape[0] == a.shape[0]
    assert n.shape[1] == 1
    assert np.allclose(n.detach().numpy(), [[2], [1/np.sqrt(2)*(4+np.cos(0))]])
    n = normal_derivative(output, b, normals)
    assert n.shape[0] == b.shape[0]
    assert n.shape[1] == 1
    assert np.allclose(n.detach().numpy(), [[3], [27/np.sqrt(2)]])


# Test divergence
def div_function(x):
    return x**2


def test_div_one_input():
    a = torch.tensor([[1.0, 0]], requires_grad=True)
    output = div_function(a)
    d = div(output, a)
    assert d.shape == (1, 1)
    d = d.detach().numpy()
    assert d[0] == 2


def test_div_many_inputs():
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0]], requires_grad=True)
    output = div_function(a)
    d = div(output, a)
    assert d.shape == (2, 1)
    d = d.detach().numpy()
    assert d[0] == 4
    assert d[1] == 6


def test_div_in_3D():
    a = torch.tensor([[1.0, 1.0, 2.0], [2.0, 1.0, 0]], requires_grad=True)
    output = div_function(a)
    d = div(output, a)
    assert d.shape[0] == 2
    assert d.shape[1] == 1
    d = d.detach().numpy()
    assert d[0] == 8
    assert d[1] == 6


def test_div_for_complexer_function_1():
    def f(x):
        out = x**2
        out[:, :1] *= x[:, 1:]
        return out
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0], [5.0, 2.0]], requires_grad=True)
    output = f(a)
    d = div(output, a)
    assert d.shape[0] == 3
    assert d.shape[1] == 1
    d = d.detach().numpy()
    assert d[0] == 4
    assert d[1] == 6
    assert d[2] == 24


def test_div_for_complexer_function_2():
    def f(x):
        out = x**2
        out[:, :1] = torch.sin(x[:, 1:] * x[:, :1])
        return out
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0], [5.0, 2.0]], requires_grad=True)
    output = f(a)
    d = div(output, a)
    assert d.shape[0] == 3
    assert d.shape[1] == 1
    d = d.detach().numpy()
    assert np.isclose(d[0], 1*(2+np.cos(1)))
    assert np.isclose(d[1], 1*(2+np.cos(2)))
    assert np.isclose(d[2], 2*(2+np.cos(10)))


# Test Jacobi-Matrix
def jac_function(x):
    out = x**2
    out[:, :1] += x[:, 1:2] 
    return out


def test_jac_one_input():
    a = torch.tensor([[1.0, 1.0]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (1, 2, 2)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[2, 1], [0, 2]]).all()


def test_jac_many_inputs():
    a = torch.tensor([[1.0, 1.0], [2.0, 1.0], [0.0, 3]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (3, 2, 2)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[2, 1], [0, 2]]).all()
    assert np.isclose(d[1], [[4, 1], [0, 2]]).all()
    assert np.isclose(d[2], [[0, 1], [0, 6]]).all()


def test_jac_in_3D():
    a = torch.tensor([[1.0, 1.0, 0.0], [2.0, 1.0, 2.0]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (2, 3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[2, 1, 0], [0, 2, 0], [0, 0, 0]]).all()
    assert np.isclose(d[1], [[4, 1, 0], [0, 2, 0], [0, 0, 4]]).all()


def test_jac_for_complexer_function():
    def jac_function(x):
        out = x**3
        out[:,:1] += torch.sin(x[:,1:2])
        out[:,1:2] *= x[:,2:]
        return out
    a = torch.tensor([[1.0, 1.0, 2.0], [2.0, 1.0, 3.0]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (2, 3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[3,  np.cos(1), 0], [0, 6, 1], [0, 0, 12]]).all()
    assert np.isclose(d[1], [[12, np.cos(1), 0], [0, 9, 1], [0, 0, 27]]).all()


def test_jac_for_complexer_function_2():
    def jac_function(x):
        out = x**2
        out[:,:1] += torch.sin(x[:,1:2]*x[:,2:])
        out[:,1:2] *= x[:,2:]
        out[:,2:] *= torch.exp(x[:,:1])
        return out
    a = torch.tensor([[1.0, 1.0, 2.0], [2.0, 1.0, 3.0]], requires_grad=True)
    output = jac_function(a)
    d = jac(output, a)
    assert d.shape == (2, 3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [[2,  np.cos(2)*2, np.cos(2)], [0, 4, 1],
                             [4*np.exp(1), 0, 4*np.exp(1)]]).all()
    assert np.isclose(d[1], [[4,  np.cos(3)*3, np.cos(3)], [0, 6, 1],
                             [9*np.exp(2), 0, 6*np.exp(2)]]).all()


# Test rot
def rot_function(x):
    out = torch.zeros((len(x), 3))
    out[:, :1] += x[:, 1:2] 
    out[:, 1:2] -= x[:, :1]
    return out


def test_rot_one_input():
    a = torch.tensor([[1.0, 1.0, 2.0]], requires_grad=True)
    output = rot_function(a)
    d = rot(output, a)
    assert d.shape == (1, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [0, 0, -2]).all()


def test_rot_many_inputs():
    a = torch.tensor([[1, 1, 2.0], [0, 1.0, 0], [1.0, 3.0, 4]], requires_grad=True)
    output = rot_function(a)
    d = rot(output, a)
    assert d.shape == (3, 3)
    d = d.detach().numpy()
    for i in range(3):
        assert np.isclose(d[i], [0, 0, -2]).all()


def test_rot_for_complexer_function():
    def rot_function(x):
        out = torch.zeros((len(x), 3))
        out[:, 1:2] -= x[:, :1]**2
        return out       
    a = torch.tensor([[-1, 1, 2.0], [1.0, 1.0, 0], [2.0, 3.0, 4]], requires_grad=True)
    output = rot_function(a)
    d = rot(output, a)
    assert d.shape == (3, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [0, 0, 2]).all()
    assert np.isclose(d[1], [0, 0, -2]).all()
    assert np.isclose(d[2], [0, 0, -4]).all()


def test_rot_for_complexer_function_2():
    def rot_function(x):
        out = torch.zeros((len(x), 3))
        out[:, :1]  = torch.sin(x[:,1:2]*x[:,2:])
        out[:, 1:2] = -x[:,:1]**2
        out[:, 2:]  = x[:,:1] + x[:,1:2]
        return out       
    a = torch.tensor([[-1, 1, 2.0], [1.0, 1.0, 0]], requires_grad=True)
    output = rot_function(a)
    d = rot(output, a)
    assert d.shape == (2, 3)
    d = d.detach().numpy()
    assert np.isclose(d[0], [1, np.cos(2)-1, 2-2*np.cos(2)]).all()
    assert np.isclose(d[1], [1, np.cos(0)-1, -2]).all()