import os
i = 0 
name = 'tilak_rinait'
for file in os.listdir('testfaces'):
	os.rename("testfaces/" + str(file), "%d.jpg" % i)
	i += 1
