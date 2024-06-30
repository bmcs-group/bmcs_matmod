import bmcs_utils.api as bu
import traits.api as tr

class GSMResponse(bu.Model):

    vars = tr.List

    record = tr.Dict

    def record_step(self, **values):
        
        for key, value in values.items():
            self.record[key].append(value)