for i in {1..21}; do
	aria2c -x 16 -s 16 -c \
	"https://zenodo.org/records/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.${i}.zip?download=1" \
	-o TAU_audio_${i}.zip
done
