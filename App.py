import Tkinter as tk

class Application(tk.Frame):

	def __init__(self, master=None):

		tk.Frame.__init__(self, master)

		self.pack()

		self.createWidgets()


	def createWidgets(self):

		self.intro = tk.Button(self)

		self.intro["text"] = "Introduce"

		self.intro["command"] = self.say_hi

		self.intro.pack(side="left")

		self.train = tk.Button(self, text="Train", fg="red", command=root.destroy)

		self.train.pack(side="left")


	def say_hi(self):

		print "Hi there everyone!"

root = tk.Tk()

app = Application(master=root)

app.mainloop()