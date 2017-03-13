from optparse import OptionParser
import sys

import datetime
from datetime import datetime as dt

from consider import Consider

import numpy as np




results = []

next1 = 0

def intro():

	print "\nHello, and welcome to MentalPrint\n"

	next1 = raw_input("Testing will begin soon, so make sure your headset is connected and transmitting properly. Continue? (y/n)")

	return next1


def write_results(contents):

	file_path = "MTest.csv"

	array = np.asarray(contents, dtype=np.float32)

	np.savetxt(file_path, array, fmt='%.2f', delimiter=',')




def record(out, session):
    con = Consider()


    if next1 == "y":

    	print "Very good, we'll monitor your brain activity for 2 minutes, and then try to identify you\n"

    else:

    	raw_input("Please press enter to begin when you are ready")


    time = 120

    for p in con.packet_generator():


        if p.attention == 0 and p.meditation == 0:
            print "Please check your headset connection\n"


        #data = map(str, [p.delta, p.theta, p.low_alpha, p.high_alpha, p.low_beta, p.high_beta, p.low_gamma, p.high_gamma, p.attention, p.meditation, dt.now().strftime('%H:%M:%S')])

        data = map(str, [p.delta, p.theta, p.low_alpha, p.high_alpha, p.low_beta, p.high_beta, p.low_gamma, p.high_gamma, p.attention, p.meditation])


        num_samples = len(results)

        time_remaining = (time - num_samples)/60

        if num_samples > 0:

        	write_results(results)

        if num_samples > time:
        	

        	raw_input("Training has concluded, open MTest.csv to try identification")

        	write_results(results)

        else:

            if p.attention != 0 or p.meditation != 0:

                results.append(data)

        if num_samples == 0:

        	print 'Training is scheduled to continue for {} more minutes\n'.format(time_remaining)

        elif num_samples > 0 and (((time % num_samples) == 0 or (time % num_samples) == 60) and (((60 + num_samples) % 60) == 0)):

            print 'Training is scheduled to continue for {} more minutes\n'.format(time_remaining)






def main():

	out = sys.stdout

	next1 = intro()

	try:

		record(out, next1)

	except KeyboardInterrupt:

		if hasattr(out, 'close'):
			out.close()




if __name__ == '__main__':
    main()
