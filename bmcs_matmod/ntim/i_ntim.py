
import traits.api as tr

class INTIM(tr.Interface):
    """
    Interface of a constitutive model at the microplane.
    """
    def get_corr_pred(self, eps_Ema, t_n1, **state):
        pass

    def get_eps_NT_p(self, **Eps):
        pass
