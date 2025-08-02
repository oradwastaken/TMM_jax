from enum import Enum


class Polarization(Enum):
    """Enum type that represents the polarization of light, either TM/p/pi or TE/s/sigma."""

    p = "p"
    s = "s"
    TM = "p"
    TE = "s"
    pi = "p"
    sigma = "s"

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(position_string):
        conversion_table = {
            "p": Polarization.p,
            "s": Polarization.s,
            "TE": Polarization.TE,
            "TM": Polarization.TM,
            "sigma": Polarization.sigma,
            "pi": Polarization.pi,
        }
        return conversion_table[position_string]
