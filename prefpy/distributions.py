# Exponential Family Distribution classes
import numpy as np

class ExponentialFamilyDistribution:
    """
    Exponential Family probability distributions have
    a PDF of the form (where x is the argument and
    theta parameterizes the distribution):
    
    f(x; theta) = exp[eta(theta)*T(x) - A(theta) + B(x)]
    
    Thus, this abstract class provides the functions:
    1. eta(theta)
    2. T(x)
    3. A(theta)
    4. B(x)
    """
    @staticmethod
    def eta(theta):
        raise NotImplementedError("abstract class must be extended")

    @staticmethod
    def T(x):
        raise NotImplementedError("abstract class must be extended")

    @staticmethod
    def A(theta):
        raise NotImplementedError("abstract class must be extended")

    @staticmethod
    def B(x):
        raise NotImplementedError("abstract class must be extended")
    

class FixedVarianceNormalDistribution(ExponentialFamilyDistribution):
    """Normal Distribution with known variance, sigma^2 = 1, sigma = 1
    Assumes parameters are mu and sigma (rather than sigma^2)
    """
    @staticmethod
    def eta(theta):
        return theta

    @staticmethod
    def T(x):
        return x

    @staticmethod
    def A(theta):
        return (theta**2)/2

    @staticmethod
    def B(x):
        return -((x**2 + np.log(2 * np.pi)) / 2)
        

class NormalDistribution(ExponentialFamilyDistribution):
    """
    Normal Distribution with unknown variance, sigma^2
    Assumes parameters are mu and sigma (rather than sigma^2)
    """
    @staticmethod
    def eta(theta):
        return np.array([theta[0] / theta[1]**2,
                         -1/(2*theta[1]**2)])

    @staticmethod
    def T(x):
        return np.array([x, x**2])

    @staticmethod
    def A(theta):
        return (theta[0]**2 / (2 * theta[1]**2)) + np.log(theta[1])

    @staticmethod
    def B(x):
        return -(np.log(2*np.pi)/2)
