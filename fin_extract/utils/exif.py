"""
Module: exif.py
Authors: Christian Bergler, Alexander Gebhard
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 25.01.2022
"""

import exifread

from GPSPhoto import gpsphoto

from PIL import Image, ExifTags


def get_decimal_from_dms(dms, ref):
	raw_string = dms.replace("'", "").replace("[", '').replace("]", '').split(',')
	gps = list()
	for s in raw_string:
		s = s.strip()
		gps.append(s)

	minutes_tmp = gps[1].split("/")

	degrees = float(gps[0])
	minutes = float(minutes_tmp[0]) / float(minutes_tmp[1]) / 60
	seconds = float(gps[2]) / 3600

	if ref in ['S', 'W']:
		degrees = -degrees
		minutes = -minutes
		seconds = -seconds

	return round(degrees + minutes + seconds, 5)


def get_gps(img_path):

	try:
		data = gpsphoto.getGPSData(img_path)
		gps_data = {'Latitude': round(data['Latitude'], 5), 'Longitude': round(data['Longitude'], 5)}
	except:
		with open(img_path, 'rb') as f:
			tags = exifread.process_file(f)

			lat = str(tags['GPS GPSLatitude'])
			lat_ref = str(tags['GPS GPSLatitudeRef'])
			long = str(tags['GPS GPSLongitude'])
			long_ref = str(tags['GPS GPSLongitudeRef'])

			lat_coord = get_decimal_from_dms(lat, lat_ref)
			long_coord = get_decimal_from_dms(long, long_ref)

			gps_data = {'Latitude': lat_coord, 'Longitude': long_coord}

	return gps_data


def get_exif_data(img_path):
	filename = img_path.replace('/', '\\').split('\\')[-1]
	img = Image.open(img_path)

	try:
		exif = {
			ExifTags.TAGS[k]: v
			for k, v in img._getexif().items()
			if k in ExifTags.TAGS
		}
	except:
		try:
			artist = filename.split("_")[2]
			if not all(x.isalpha() or x.isspace() for x in artist):
				artist = 'None'
		except:
			artist = 'None'
		return (filename, artist, 'None', 'None')

	#artist
	try:
		artist = exif['Artist'].rstrip()

		if artist == '' or artist == 'None' or artist is None:
			artist = filename.split("_")[2]
			if not all(x.isalpha() or x.isspace() for x in artist):
				artist = 'None'
	except:
		try:
			artist = filename.split("_")[2]
			if not all(x.isalpha() or x.isspace() for x in artist):
				artist = 'None'
		except:
			artist = 'None'
	#GPS
	try:
		gps = get_gps(img_path)
	except:
		gps = 'None'

	#date
	try:
		date = exif['DateTimeOriginal']
	except:
		date = 'None'

	return filename, artist, gps, date

