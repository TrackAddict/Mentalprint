from optparse import OptionParser
import sys

import datetime
from datetime import datetime as dt

from consider import Consider

import numpy as np


results = []

#TODO: Write on each sampling

def intro():

	print "\nThank you for providing your MentalPrint\n"

	name = raw_input("Please type your name here: ")

	print "\nThank you " + name + "\n"

	session =  int(raw_input("How long (in minutes) will you be able to record for? "))

	next1 = raw_input("Please make sure your headset is connected and transmitting properly. Continue? (y/n)")

	return name, session, next1



# def write_results(file_name, contents):
# 	file_path = file_name + ".csv"

# 	with open(file_path, "a") as this_file:

# 		this_file.write(str(contents) + "\n")

def write_results(file_name, contents):

	#format = ('%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s')

	file_path = file_name + ".csv"

	array = np.asarray(contents, dtype=np.float32)

	np.savetxt(file_path, array, fmt='%.2f', delimiter=',')




def record(out, name, session, next1):
    con = Consider()


    if next1 == "y":

    	print "\n"

    else:

    	raw_input("Please press enter to begin when you are ready")


    time = session * 60

    for p in con.packet_generator():


        if p.attention == 0 and p.meditation == 0:
            print "Please check your headset connection\n"


        #data = map(str, [p.delta, p.theta, p.low_alpha, p.high_alpha, p.low_beta, p.high_beta, p.low_gamma, p.high_gamma, p.attention, p.meditation, dt.now().strftime('%H:%M:%S')])

        data = map(str, [p.delta, p.theta, p.low_alpha, p.high_alpha, p.low_beta, p.high_beta, p.low_gamma, p.high_gamma, p.attention, p.meditation])


        num_samples = len(results)

        time_remaining = (time - num_samples)/60

        if num_samples > 0:

        	write_results(name, results)

        if num_samples > time:

        	write_results(name, results)

        	raw_input("Training has concluded, you may remove the headset at your leisure")

        	break

        else:

            if p.attention != 0 or p.meditation != 0:

                results.append(data)

        if num_samples > 0 and (((time % num_samples) == 0 or (time % num_samples) == 60) and (((60 + num_samples) % 60) == 0)):

            print 'Training is scheduled to continue for {} more minutes\n'.format(time_remaining)




def main():

	out = sys.stdout

	name, session, next1 = intro()

	try:

		print "Training in progress...\n"

		record(out, name, session, next1)

	except KeyboardInterrupt:

		write_results(name, results)

		if hasattr(out, 'close'):
			out.close()



if __name__ == '__main__':
    main()
