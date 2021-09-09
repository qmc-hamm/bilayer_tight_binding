import numpy as np 

def exponential(x, a, b):
    return a * np.exp(-b * (x - 6.33))

def moon(r, a, b, c): 
    """
    Parameterization from Moon and Koshino, Phys. Rev. B 85, 195458 (2012)
    """
    d, dz = r 
    return a * np.exp(-b * (d - 2.68))*(1 - (dz/d)**2) + c * np.exp(-b * (d - 6.33)) * (dz/d)**2

def fang(rvec, a0, b0, c0, a3, b3, c3, a6, b6, c6, d6):
    """
    Parameterization from Fang and Kaxiras, Phys. Rev. B 93, 235153 (2016)
    """
    r, theta12, theta21 = rvec
    r = r / 4.649 

    def v0(x, a, b, c): 
        return a * np.exp(-b * x ** 2) * np.cos(c * x)

    def v3(x, a, b, c): 
        return a * (x ** 2) * np.exp(-b * (x - c) ** 2)  

    def v6(x, a, b, c, d): 
        return a * np.exp(-b * (x - c)**2) * np.sin(d * x)

    f =  v0(r, a0, b0, c0) 
    f += v3(r, a3, b3, c3) * (np.cos(3 * theta12) + np.cos(3 * theta21))
    f += v6(r, a6, b6, c6, d6) * (np.cos(6 * theta12) + np.cos(6 * theta21))
    return f
