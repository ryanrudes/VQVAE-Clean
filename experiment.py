from goexplore.algorithm import GoExplore
from rich.console import Console
from getpass import getpass
from rich import print
from time import time
import mail
import json
import os

class Experiment:
    def __init__(self, env, **kwargs):
        self.goexplore = GoExplore(env)
        self.kwargs = kwargs

    def agreement(self, start, address):
        console = Console()
        console.rule("[bold red]IMPORTANT NOTICE", style = "red")
        console.print(" [bold blue]General Information")
        console.print("")
        console.print(" • Following this experiment, an email with subject title '%s' will be sent to '%s'" % (start, address))
        console.print(" • In order to protect your privacy, [underline]at no point during this experiment will your password be recorded, nor will it be printed to the console.")
        console.print(" • The results logged will be sent via a secure connection over port 587")
        console.print(" • Results will be saved to 'experiments/%s/results.json'" % start)
        console.print(" ")
        console.print(" [bold blue]You must enable less secure app access")
        console.print(" ")
        console.print(" • Enable less secure app access [link]https://myaccount.google.com/lesssecureapps")
        console.print(" ")
        console.print(" [bold blue]If you have 2-step verification enabled, you must either disable it or create an app-specific password for less secure apps")
        console.print(" ")
        console.print(" • Create an app password        [link]https://support.google.com/mail/answer/185833")
        console.print(" • Disable 2-step verification   [link]https://myaccount.google.com/signinoptions/two-step-verification")
        console.print(" ")
        console.print(" Read the above notice. Then, press any key to proceed[blink]...")
        console.rule(style = "red")
        input(' ')
        return getpass(" Enter email password to continue: ")

    def run(self, duration, experiments, record=[], sendmail=False, address='', callback=lambda _: {}):
        if not os.path.exists('experiments'):
            os.mkdir('experiments')

        start = input('Enter a name for this experiment (or press enter to use current system time): ')
        if not start:
            start = str(time())

        if sendmail:
            mail.validate(address)
            password = self.agreement(start, address)

        save = f'experiments/{start}'
        os.mkdir(save)

        results = {'experiments': {}}
        for experiment in range(experiments):
            self.goexplore.initialize(**self.kwargs)
            self.goexplore.run_for(duration, desc = f'Experimenting... ({experiment + 1}/{experiments})', **self.kwargs)
            results['experiments'][experiment] = {}
            for attr in record:
                results['experiments'][experiment][attr] = getattr(self.goexplore, attr)
            results.update(callback(self.goexplore, experiment, save))

        if sendmail:
            message = json.dumps(results, indent = 4)
            mail.send(message, address, password, start)

        with open(os.path.join(save, 'results.json'), 'w') as f:
            json.dump(results, f)
