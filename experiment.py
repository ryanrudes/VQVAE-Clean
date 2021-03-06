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

    def agreement(self, title, address, password):
        console = Console()
        console.rule("[bold red]IMPORTANT NOTICE", style = "red")
        console.print(" [bold blue]General Information")
        console.print("")
        console.print(" • Following this experiment, an email with subject title '%s' will be sent to '%s'" % (title, address))
        console.print(" • In order to protect your privacy, [underline]at no point during this experiment will your password be recorded, nor will it be printed to the console.")
        console.print(" • The results logged will be sent via a secure connection over port 587")
        console.print(" • Results will be saved to 'experiments/%s/results.json'" % title)
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
        if password is None:
            return getpass(" Enter email password to continue: ")
        else:
            return password

    def run(self, duration, experiments, record=[], sendmail=False, address='', password=None, callback=lambda _: {}, title=None, showinfo=True):
        if not os.path.exists('experiments'):
            os.mkdir('experiments')

        if title is None:
            title = input('Enter a name for this experiment (or press enter to use current system time): ')

        if title:
            save = f'experiments/{title}'
            if os.path.exists(save):
                duplicateID = 2
                temp = f'{save} ({duplicateID})'
                while os.path.exists(temp):
                    duplicateID += 1
                    temp = f'{save} ({duplicateID})'
                save = temp
        else:
            title = str(time())
            save = f'experiments/{title}'

        os.mkdir(save)

        if sendmail:
            mail.validate(address)
            if showinfo:
                password = self.agreement(title, address, password)

        results = {'experiments': {}}
        for experiment in range(experiments):
            self.goexplore.initialize(**self.kwargs)
            self.goexplore.run_for(duration, desc = f'Experimenting... ({experiment + 1}/{experiments})', **self.kwargs)
            results['experiments'][experiment] = {}
            for attr in record:
                results['experiments'][experiment][attr] = getattr(self.goexplore, attr)
            results['experiments'][experiment].update(callback(self.goexplore, experiment, save))

        if sendmail:
            message = json.dumps(results, indent = 4)
            mail.send(message, address, password, title)

        with open(os.path.join(save, 'results.json'), 'w') as f:
            json.dump(results, f)

        self.goexplore.close()
