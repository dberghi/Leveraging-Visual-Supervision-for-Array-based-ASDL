You can decide which supervisory condition to use by changing the 'supervisory_condition' in config.py
This will change which development csv file to employ. Naming convention as development_[location pseudo-labels]_[VA].csv

The location pseudo-labels are achieved with
	TalkNet: https://github.com/TaoRuijie/TalkNet-ASD
	or
	ASC: https://github.com/fuankarion/active-speakers-context

The automatic voice activity (VA) labels are achieved with a VAD
	WebRTC: https://github.com/wiseman/py-webrtcvad

otherwise labels come from the grund truth (GT) labels provided by TragicTalkers: https://cvssp.org/data/TragicTalkers/
