import argparse


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """__call__ _summary_

        Args:
            parser (_type_): _description_
            namespace (_type_): _description_
            values (_type_): _description_
            option_string (_type_, optional): _description_. Defaults to None.
        """
        setattr(namespace, self.dest, dict())
        values = values.rsplit(",")
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def parse_shell():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out")
    parser.add_argument("-d", "--data")
    parser.add_argument("-p", "--pheno")
    parser.add_argument("-s", "--snps", nargs="*")
    parser.add_argument("-u", "--utils")
    parser.add_argument("-w", "--wild", action=ParseKwargs)
    args = parser.parse_intermixed_args()

    return args