# Nicholas Falter, James Mare', Riley Ruckman, Travis Shields
# Senior Project
# Labeler for Log Files


from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import json
from typing import Type


labels = []
counter = 1

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
#cwd = os.getcwd()
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

with open(filename, 'r', encoding='utf8') as logs:
    for message in logs:
        message = message.strip('\n')
        # An error ssh-related error message appears when too many sign-in attempts have been tried. This is checked for and automatically labeled.
        if 'sshd[' in message and 'error' in message:
            label = "ssh"
        # Any other ssh message will be safe.
        elif 'sshd[' in message:
            label = 'safe'
        # 'sudo apt autoremove' and 'sudo apt-get autoremove' are flagged as a bad actor.
        elif 'sudo' in message and 'autoremove' in message:
            label = "su"
        # The code 'AH01618' occurs when an invalid username is entered. This situation is marked as a bad actor.
        elif 'AH01618' in message:
            label = 'ws'
        # The code 'AH01617' occurs when an invalid password is entered.
        #elif 'AH01617' in message:
            #label = 'ws'
        # The user enters the appropriate label for the current message
        else:
            print("{} Log Message: {}".format(counter, message))
            label = input("Enter label: ")

        labels.append(label)

        counter += 1

# Grabs the name of the log file to use in creating the file for its labels.
filename = str(filename).split('/')[-1].strip('.txt')

if len(labels):
    with open("{}_labels.txt".format(filename), 'w', encoding='utf8') as file:
        json.dump(labels, file)
