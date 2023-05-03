Code for paper "Search-Map-Search: A New Frame Selection Paradigm for Action Recognition" adapted from MMAction2 (https://github.com/open-mmlab/mmaction2)


	sms/feature_extraction.py 
	    
	    Extract features


	sms/search_gls.py

	    Search the best combinations with hierarchical GLS


	sms/train_mapping.py

	    Train feature mapping function


	sms/infer_mapping.py

	    Infer frame combinations with trained mapping function and search



Results

	selected_frames/

		Contains the frame selection results with SMS for ActivityNet, FCVID, and UCF101