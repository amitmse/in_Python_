import win32com.client
import os

#https://stackoverflow.com/questions/40849742/outlook-using-python-win32com-to-iterate-subfolders
#https://hackerpython.wordpress.com/2015/09/29/how-to-download-email-attachments-from-outlook/
#https://stackoverflow.com/questions/23638963/trying-to-save-an-attachment-from-outlook-via-pythoncom
#https://stackoverflow.com/questions/43877606/python-script-to-download-attachments-in-email-having-keywords-in-subject-line
		
outlook = win32com.client.Dispatch("Outlook.Application")
namespace = outlook.GetNamespace("MAPI")
root_folder = namespace.Folders.Item(1)
subfolder = root_folder.Folders['Inbox'].Folders['Work'].Folders['Dish']
messages = subfolder.Items		
'''
outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
inbox = outlook.GetDefaultFolder(6)
messages = inbox.Items
message = messages.GetFirst()
'''
while True:
    try:
        print (message)
        attachments = message.Attachments
        attachment = attachments.Item(1)
        attachment.SaveASFile(os.getcwd() + '\\email\\' + str(attachment))
        print (attachment)
        message = messages.GetNext()
    except:
        message = messages.GetNext()
