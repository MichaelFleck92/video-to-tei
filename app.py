import json
import boto3
import time
import requests
import mimetypes
from lxml import etree
from datetime import datetime

'''
    script configuration variables start
'''

file_path = "/path/to/filedir/"
file_name = "filename.mp4"
tei_output_file_name = "filename.xml"

# Spoken language in the recording
language_code = "de-DE" # Language code based on aws3.link/languagecodes
analyze_source_language = "de" # ISO 639-1
translation_target_language = "en" # ISO 639-1

# Create a custom vocabulary beforehand in AWS Transcribe console if wanted, keep variable string empty if not used
custom_vocabulary = ""

aws_region_name = ""
aws_access_key_id = ""
aws_secret_access_key = ""

# S3 Bucket Name to store recording
s3_bucket_name = ""

'''
    script configuration variables end
'''

'''
    global variable declaration start
'''
transcribe_client = boto3.client('transcribe',
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id, 
    aws_secret_access_key=aws_secret_access_key,
)
s3_client = boto3.client('s3',
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id, 
    aws_secret_access_key=aws_secret_access_key,
)
translate_client = boto3.client('translate',
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id, 
    aws_secret_access_key=aws_secret_access_key,
)
comprehend_client = boto3.client('comprehend',
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id, 
    aws_secret_access_key=aws_secret_access_key,
)
rekognition_client = boto3.client('rekognition',
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id, 
    aws_secret_access_key=aws_secret_access_key,
)

unix_timestamp = str(int(time.time()))
s3_key = unix_timestamp+"_"+file_name

annotation_dict = {}

'''
    global variable declaration end
'''



def startAnnotationJobs():
    
    print("Start uploading file")
    with open(file_path+file_name, 'rb') as data:
        s3_client.upload_fileobj(data, s3_bucket_name, s3_key)
    s3_uri = "s3://"+s3_bucket_name+"/"+s3_key
    print("File uploaded to S3")

    
    settings = {}
    if custom_vocabulary:
        settings["VocabularyName"] = custom_vocabulary
    print("Start transcription job")
    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=s3_key,
        LanguageCode=language_code,
        Media={
            'MediaFileUri': s3_uri,
        },
        Settings=settings,
        #ModelSettings={
        #    'LanguageModelName': 'string'
        #},
    )


    print("Start video analyzing jobs")
    label_detection_response = rekognition_client.start_label_detection(
        Video={
            'S3Object': {
                'Bucket': s3_bucket_name,
                'Name': s3_key,
            }
        },
        MinConfidence=90,
    )
    label_detection_response_jobId = label_detection_response["JobId"]
    #print(label_detection_response["JobId"])
    segment_detection_response = rekognition_client.start_segment_detection(
        Video={
            'S3Object': {
                'Bucket': s3_bucket_name,
                'Name': s3_key,
            }
        },
        Filters={
            'TechnicalCueFilter': {
                'MinSegmentConfidence': 90
            },
            'ShotFilter': {
                'MinSegmentConfidence': 90
            }
        },
        SegmentTypes=[
            'TECHNICAL_CUE', 'SHOT'
        ]
    )
    segment_detection_response_jobId = segment_detection_response["JobId"]
    #print(segment_detection_response["JobId"])
    text_detection_response = rekognition_client.start_text_detection(
        Video={
            'S3Object': {
                'Bucket': s3_bucket_name,
                'Name': s3_key,
            }
        },
        Filters={
            'WordFilter': {
                'MinConfidence': 90,
                'MinBoundingBoxHeight': 0.05,
                'MinBoundingBoxWidth': 0.05
            },
        }
    )
    text_detection_response_jobId = text_detection_response["JobId"]
    #print(text_detection_response["JobId"])
    celebrity_detection_response = rekognition_client.start_celebrity_recognition(
        Video={
            'S3Object': {
                'Bucket': s3_bucket_name,
                'Name': s3_key,
            }
        },
    )
    celebrity_detection_response_jobId = celebrity_detection_response["JobId"]
    #print(celebrity_detection_response["JobId"])



    print("Waiting for transcription job to be complete - retry every 15 seconds")
    while (response["TranscriptionJob"]["TranscriptionJobStatus"] != "COMPLETED") and (response["TranscriptionJob"]["TranscriptionJobStatus"] != "FAILED"):
        time.sleep(15)

        response = transcribe_client.get_transcription_job(
            TranscriptionJobName=s3_key
        )

        if response["TranscriptionJob"]["TranscriptionJobStatus"] == "FAILED":
            raise Exception("Transcription job failed")

        print("Transcription job not finished")

    
    print("Transcription job finished")

    url = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    r = requests.get(url, allow_redirects=True)
    open('raw_transcript.json', 'wb').write(r.content)
    
    with open('raw_transcript.json') as f_in:
        transcript_json = json.load(f_in)

    utterances = []
    previous_end_time = float(transcript_json["results"]["items"][0]["start_time"])
    previous_start_time = 0
    utterance = ""
    utterance_start = 0
    utterance_end = 0
    # Finding last element with an end_time
    index = 0
    while True:
        try:
            index = index - 1
            utterance_end = float(transcript_json["results"]["items"][index]["end_time"])
            break
        except KeyError:
            continue
    duration = 0
    for item in transcript_json["results"]["items"]:
        if item["type"] == "pronunciation":
            difference = float(item["start_time"]) - previous_end_time
            if utterance_start == 0:
                utterance_start = item["start_time"]
            if difference > 3:
                utterance_end = previous_end_time
                duration = float(utterance_end) - float(utterance_start)
                # Handle case if utterance only has a single word
                if duration < 0:
                    utterance_start = str(previous_start_time)
                duration = float(utterance_end) - float(utterance_start)
                #print("\n\n"+utterance)
                utterances.append({
                    'text': utterance.lstrip(),
                    'start': utterance_start,
                    'end': str(previous_end_time),
                    'dur': '%.2f' % duration
                })
                utterance = ""
                utterance_start = 0
                utterance_end = 0
                utterance = utterance + " " + item["alternatives"][0]["content"]
            else:
                utterance = utterance + " " + item["alternatives"][0]["content"]
            previous_end_time = float(item["end_time"])
            previous_start_time = float(item["start_time"])
        elif item["type"] == "punctuation":
            utterance = utterance + item["alternatives"][0]["content"]
    #print("\n\n"+utterance)
    duration = float(previous_end_time) - float(utterance_start)
    utterances.append({
        'text': utterance.lstrip(),
        'start': utterance_start,
        'end': str(previous_end_time),
        'dur': '%.2f' % duration
    })
    

    print("Running translation, entity, sentiment and syntax recognition")
    for utterance in utterances:
        
        # Translation
        response = translate_client.translate_text(
            Text=utterance["text"],
            SourceLanguageCode=analyze_source_language,
            TargetLanguageCode=translation_target_language,
        )
        utterance["translation"] = response["TranslatedText"]
        
        # Entities
        response = comprehend_client.detect_entities(
            Text=utterance["text"],
            LanguageCode=analyze_source_language,
        )
        entities = []
        for entity in response["Entities"]:
            if entity["Score"] > 0.7:
                entities.append(entity)
        utterance["entities"] = entities

        # Sentiment
        response = comprehend_client.detect_sentiment(
            Text=utterance["text"],
            LanguageCode=analyze_source_language,
        )
        utterance["sentiment"] = response["Sentiment"]
        
        response = comprehend_client.detect_syntax(
            Text=utterance["text"],
            LanguageCode='de',
        )
        utterance["syntax"] = response["SyntaxTokens"]

    annotation_dict['utterances'] = utterances
    


    # Wait for all video analyzing jobs to be finished
    print("Waiting for all video analyzing jobs to be complete")
    label_finished = False
    segment_finished = False
    text_finished = False
    celebrity_finished = False
    
    while (label_finished is False) or (segment_finished is False) or (text_finished is False) or (celebrity_finished is False):

        label_detection_response = rekognition_client.get_label_detection(
            JobId=label_detection_response_jobId,
        )
        if label_detection_response["JobStatus"] == "SUCCEEDED":
            label_finished = True
        
        segment_detection_response = rekognition_client.get_segment_detection(
            JobId=segment_detection_response_jobId,
        )
        if segment_detection_response["JobStatus"] == "SUCCEEDED":
            segment_finished = True
        
        text_detection_response = rekognition_client.get_text_detection(
            JobId=text_detection_response_jobId,
        )
        if text_detection_response["JobStatus"] == "SUCCEEDED":
            text_finished = True
        
        celebrity_detection_response = rekognition_client.get_celebrity_recognition(
            JobId=celebrity_detection_response_jobId,
        )
        if celebrity_detection_response["JobStatus"] == "SUCCEEDED":
            celebrity_finished = True

        if (label_detection_response["JobStatus"] == "FAILED") or (segment_detection_response["JobStatus"] == "FAILED") or (text_detection_response["JobStatus"] == "FAILED") or (celebrity_detection_response["JobStatus"] == "FAILED"):
            raise Exception("Video analyzing jobs failed")

        print("Video analyzing jobs not finished - retry in 15 seconds")

        time.sleep(15)
    
    
    # Save AWS responses to json files
    with open('raw_label_detection_response.json', 'w') as fp:
        json.dump(label_detection_response, fp)

    with open('raw_segment_detection_response.json', 'w') as fp:
        json.dump(segment_detection_response, fp)

    with open('raw_text_detection_response.json', 'w') as fp:
        json.dump(text_detection_response, fp)

    with open('raw_celebrity_detection_response.json', 'w') as fp:
        json.dump(celebrity_detection_response, fp)
    
    
    print("Analyzing text, segments, labels and celebrities")

    with open('raw_text_detection_response.json') as f_in:
        text_detection_response = json.load(f_in)

    detected_text_array = []
    for detection in text_detection_response["TextDetections"]:
        detected_text = {}
        if detection["TextDetection"]["Type"] == "LINE":
            detected_text["detected_text"] = detection["TextDetection"]["DetectedText"]
            timestamp = float(detection["Timestamp"])/1000.0
            detected_text["timestamp"] = "%.2f" % timestamp
            #print(detection["TextDetection"]["DetectedText"])
            detected_text_array.append(detected_text)
    #print(detected_text_array)
    annotation_dict['detected_text'] = detected_text_array
    

    with open('raw_segment_detection_response.json') as f_in:
        segment_detection_response = json.load(f_in)

    cue_detection_array = []
    shot_detection_array = []
    for segment in segment_detection_response["Segments"]:
        if segment["Type"] == "TECHNICAL_CUE":
            detected_cue = {}
            timestamp = float(segment["StartTimestampMillis"])/1000.0
            detected_cue["start"] = "%.2f" % timestamp
            timestamp = float(segment["EndTimestampMillis"])/1000.0
            detected_cue["end"] = "%.2f" % timestamp
            timestamp = float(segment["DurationMillis"])/1000.0
            detected_cue["dur"] = "%.2f" % timestamp
            detected_cue["type"] = segment["TechnicalCueSegment"]["Type"]
            cue_detection_array.append(detected_cue)
        if segment["Type"] == "SHOT":
            detected_shot = {}
            timestamp = float(segment["StartTimestampMillis"])/1000.0
            detected_shot["start"] = "%.2f" % timestamp
            timestamp = float(segment["EndTimestampMillis"])/1000.0
            detected_shot["end"] = "%.2f" % timestamp
            timestamp = float(segment["DurationMillis"])/1000.0
            detected_shot["dur"] = "%.2f" % timestamp
            detected_shot["index"] = segment["ShotSegment"]["Index"]
            shot_detection_array.append(detected_shot)
    annotation_dict['detected_cues'] = cue_detection_array
    annotation_dict['detected_shots'] = shot_detection_array
    #print(shot_detection_array)
    #print(cue_detection_array)
    

    with open('raw_label_detection_response.json') as f_in:
        label_detection_response = json.load(f_in)

    label_detection_array = []
    for label in label_detection_response["Labels"]:
        detected_label = {}
        detected_label["label_name"] = label["Label"]["Name"]
        timestamp = float(label["Timestamp"])/1000.0
        detected_label["timestamp"] = "%.2f" % timestamp
        label_detection_array.append(detected_label)



    annotation_dict['detected_labels'] = label_detection_array      
    #print(label_detection_array)    
    
    with open('raw_celebrity_detection_response.json') as f_in:
        celebrity_detection_response = json.load(f_in)

    celebrity_detection_array = []
    for celebrity in celebrity_detection_response["Celebrities"]:
        if celebrity["Celebrity"]["Confidence"] > 90:
            detected_celebrity = {}
            detected_celebrity["name"] = celebrity["Celebrity"]["Name"]
            timestamp = float(label["Timestamp"])/1000.0
            detected_celebrity["timestamp"] = "%.2f" % timestamp
            urls = celebrity["Celebrity"]["Urls"]
            detected_celebrity["urls"] = urls
            celebrity_detection_array.append(detected_celebrity)

    annotation_dict['detected_celebrities'] = celebrity_detection_array  
    #print(celebrity_detection_array)



    # Metadata
    metadata = {}
    metadata["FileName"] = file_name
    metadata["Duration"] = celebrity_detection_response["VideoMetadata"]["DurationMillis"] / 1000
    metadata["Format"] = celebrity_detection_response["VideoMetadata"]["Format"]
    metadata["FrameRate"] = celebrity_detection_response["VideoMetadata"]["FrameRate"]
    metadata["FrameHeight"] = str(celebrity_detection_response["VideoMetadata"]["FrameHeight"])
    metadata["FrameWidth"] = str(celebrity_detection_response["VideoMetadata"]["FrameWidth"])
    metadata["VideoLanguage"] = analyze_source_language
    metadata["TranslationLanguage"] = translation_target_language
    annotation_dict["Metadata"] = metadata


    # Save final annotation json file
    with open('annotation.json', 'w') as fp:
        json.dump(annotation_dict, fp)

    
    return True


def increaseOffsets(transcribtDict, beginOffset, endOffset, preIncreaseValue, postIncreaseValue)  -> dict:
    for entity in transcribtDict["entities"]:
        if entity["BeginOffset"] >= endOffset:
            entity["BeginOffset"] = entity["BeginOffset"] + postIncreaseValue
            entity["EndOffset"] = entity["EndOffset"] + postIncreaseValue
        elif entity["BeginOffset"] >= beginOffset:
            entity["BeginOffset"] = entity["BeginOffset"] + preIncreaseValue
            entity["EndOffset"] = entity["EndOffset"] + preIncreaseValue
    for pos in transcribtDict["syntax"]:
        if pos["BeginOffset"] >= endOffset:
            pos["BeginOffset"] = pos["BeginOffset"] + postIncreaseValue
            pos["EndOffset"] = pos["EndOffset"] + postIncreaseValue
        elif pos["BeginOffset"] >= beginOffset:
            pos["BeginOffset"] = pos["BeginOffset"] + preIncreaseValue
            pos["EndOffset"] = pos["EndOffset"] + preIncreaseValue
    return transcribtDict
            


def createTeiFile() -> etree.Element:
    '''
        Function to parse the annotation json file to a TEI document
    '''
    with open('annotation.json') as f_in:
        transcript_json = json.load(f_in)

    TEI = etree.Element('TEI', xmlns="http://www.tei-c.org/ns/1.0")
    TEI.addprevious(etree.ProcessingInstruction("xml-model", text='href="http://www.tei-c.org/release/xml/tei/custom/schema/relaxng/tei_all.rng" type="application/xml" schematypens="http://purl.oclc.org/dsdl/schematron"'))
    TEI.addprevious(etree.ProcessingInstruction("xml-model", text='href="https://raw.githubusercontent.com/MichaelFleck92/video-to-tei/main/custom-scheme.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"'))

    teiHeader = etree.Element('teiHeader')
    fileDesc = etree.Element('fileDesc')
    titleStmt = etree.Element('titleStmt')
    title = etree.Element('title')
    title.text = transcript_json["Metadata"]["FileName"]
    titleStmt.append(title)
    publicationStmt = etree.Element('publicationStmt')
    p = etree.Element('p')
    p.text = "Publication Information"
    publicationStmt.append(p)
    sourceDesc = etree.Element('sourceDesc')
    p = etree.Element('p')
    p.text = "Automatically generated from "+transcript_json["Metadata"]["FileName"]+" by MichaelFleck92/video-to-tei"
    sourceDesc.append(p)

    fileDesc.append(titleStmt)
    fileDesc.append(publicationStmt)
    fileDesc.append(sourceDesc)







    sourceDesc = etree.Element('sourceDesc')
    recordingStmt = etree.Element('recordingStmt')
    recording = etree.Element('recording',
        type="video"
    )
    mt = mimetypes.guess_type(file_path+file_name)
    media = etree.Element('media',
        url=transcript_json["Metadata"]["FileName"],
        mimeType=mt[0]
    )
    media.attrib["{http://www.w3.org/XML/1998/namespace}lang"] = transcript_json["Metadata"]["VideoLanguage"]
    desc = etree.Element('desc')
    dimensions = etree.Element('dimensions')
    width = etree.Element('width')
    width.text = transcript_json["Metadata"]["FrameWidth"]
    dimensions.append(width)
    height = etree.Element('height')
    height.text = transcript_json["Metadata"]["FrameHeight"]
    dimensions.append(height)
    desc.append(dimensions)

    media.append(desc)
    recording.append(media)
    recordingStmt.append(recording)
    sourceDesc.append(recordingStmt)
    fileDesc.append(sourceDesc)

    teiHeader.append(fileDesc)
    TEI.append(teiHeader)

    standOff = etree.Element('standOff')

    # Get unique pers, place and orgNames
    unique_persons = set()
    unique_locations = set()
    unique_organizations = set()
    for utterance in transcript_json["utterances"]:
        for entity in utterance["entities"]:
            if entity["Type"] == "PERSON":
                unique_persons.add(entity["Text"])
            if entity["Type"] == "LOCATION":
                unique_locations.add(entity["Text"])
            if entity["Type"] == "ORGANIZATION":
                unique_organizations.add(entity["Text"])
    persons_dict = []
    unique_celebreties = []
    unique_celebreties_names = []
    counter = 1
    for celebrity in transcript_json["detected_celebrities"]:
        if celebrity["name"] not in unique_celebreties_names:
            unique_celebreties.append(celebrity)
            unique_celebreties_names.append(celebrity["name"])
            celebrity["id"] = "pers"+f'{counter:03d}'
            counter = counter + 1
    for person in unique_persons:
        persons_dict.append({
            "id": "pers"+f'{counter:03d}',
            "persName": str(person)
        })
        counter = counter + 1
    locations_dict = []
    counter = 1
    for location in unique_locations:
        locations_dict.append({
            "id": "place"+f'{counter:03d}',
            "placeName": str(location)
        })
        counter = counter + 1
    organizations_dict = []
    counter = 1
    for organization in unique_organizations:
        organizations_dict.append({
            "id": "org"+f'{counter:03d}',
            "orgName": str(organization)
        })
        counter = counter + 1

    def getReferenceIdByText(name, dictType) -> str:
        if dictType == "pers":
            for person in persons_dict:
                if person["persName"] == name:
                    return person["id"]
        elif dictType == "celebrity":
            for celebrity in unique_celebreties:
                if celebrity["name"] == name:
                    return celebrity["id"]
        elif dictType == "place":
            for place in locations_dict:
                if place["placeName"] == name:
                    return place["id"]
        elif dictType == "org":
            for org in organizations_dict:
                if org["orgName"] == name:
                    return org["id"]
        else:
            return None

    
    listPerson = etree.Element('listPerson')

    for celebrity in unique_celebreties:
        person = etree.Element('person')
        person.attrib["{http://www.w3.org/XML/1998/namespace}id"] = celebrity["id"]
        persName = etree.Element('persName')
        persName.text = celebrity["name"]
        person.append(persName)
        for url in celebrity["urls"]:
            idno = etree.Element('idno')
            idno.text = url
            person.append(idno)

        listPerson.append(person)

    
    personGrp = etree.Element('personGrp', role="determined-by-utterances")
    for person in persons_dict:
        persName = etree.Element('persName')
        persName.attrib["{http://www.w3.org/XML/1998/namespace}id"] = person["id"]
        persName.text = person["persName"]
        personGrp.append(persName)
    listPerson.append(personGrp)

    standOff.append(listPerson)

    listPlace = etree.Element('listPlace', type="determined-by-utterances")
    for place in locations_dict:
        place_tag = etree.Element('place')
        placeName = etree.Element('placeName')
        placeName.attrib["{http://www.w3.org/XML/1998/namespace}id"] = place["id"]
        placeName.text = place["placeName"]
        place_tag.append(placeName)
        listPlace.append(place_tag)
    standOff.append(listPlace)

    listOrg = etree.Element('listOrg', type="determined-by-utterances")
    for org in organizations_dict:
        org_tag = etree.Element('org')
        orgName = etree.Element('orgName')
        orgName.attrib["{http://www.w3.org/XML/1998/namespace}id"] = org["id"]
        orgName.text = org["orgName"]
        org_tag.append(orgName)
        listOrg.append(org_tag)
    standOff.append(listOrg)

    interpGrp = etree.Element('interpGrp')
    interpGrp.attrib["{http://www.w3.org/XML/1998/namespace}id"] = "sentiment"
    sentiment_array = []
    for utterance in transcript_json["utterances"]:
        if utterance["sentiment"] not in sentiment_array:
            sentiment_array.append(utterance["sentiment"])
            interp = etree.Element('interp')
            interp.attrib["{http://www.w3.org/XML/1998/namespace}id"] = utterance["sentiment"]
            interpGrp.append(interp)
    standOff.append(interpGrp)

    TEI.append(standOff)


    text = etree.Element('text')
    body = etree.Element('body')

    for cue in transcript_json["detected_cues"]:
        cue_elements_array = []
        div_cue = etree.Element('div',
            type="DetectedCue"
        )
        if cue["type"] == "OpeningCredits":
            div_cue.attrib["type"] = "OpeningCredits"
        elif cue["type"] == "Content":
            div_cue.attrib["type"] = "Content"
        elif cue["type"] == "EndCredits":
            div_cue.attrib["type"] = "EndCredits"
        div_cue.attrib["from"] = time.strftime('%H:%M:%S', time.gmtime(float(cue["start"])))
        div_cue.attrib["to"] = time.strftime('%H:%M:%S', time.gmtime(float(cue["end"])))
        div_cue.attrib["dur"] = time.strftime('%H:%M:%S', time.gmtime(float(cue["dur"])))

        cue_start = float(cue["start"])
        if cue["type"] == "OpeningCredits":
            cue_start = 0.0
        cue_end = float(cue["end"])

        for shot in transcript_json["detected_shots"]:
            if float(shot["start"]) >= cue_start and float(shot["end"]) <= cue_end:
                shot_elements_array = []
                div_shot = etree.Element('div',
                    type="DetectedShot"
                )
                div_shot.attrib["from"] = time.strftime('%H:%M:%S', time.gmtime(float(shot["start"])))
                div_shot.attrib["to"] = time.strftime('%H:%M:%S', time.gmtime(float(shot["end"])))
                div_shot.attrib["dur"] = shot["dur"]

                shot_start = float(shot["start"])
                shot_end = float(shot["end"])

                for detected_text in transcript_json["detected_text"]:
                    if float(detected_text["timestamp"]) >= shot_start and float(detected_text["timestamp"]) <= shot_end:
                        div_text = etree.Element('div',
                            type="DetectedText"
                        )
                        div_text.attrib["when"] = time.strftime('%H:%M:%S', time.gmtime(float(detected_text["timestamp"])))
                        caption = etree.Element('caption')
                        caption.text = detected_text["detected_text"]
                        div_text.append(caption)
                        shot_elements_array.append(div_text)

                for label in transcript_json["detected_labels"]:
                    if float(label["timestamp"]) >= shot_start and float(label["timestamp"]) <= shot_end:
                        div_label = etree.Element('div',
                            type="DetectedLabel"
                        )
                        div_label.attrib["when"] = time.strftime('%H:%M:%S', time.gmtime(float(label["timestamp"])))
                        ab = etree.Element('ab')
                        ab.text = label["label_name"]
                        div_label.append(ab)
                        shot_elements_array.append(div_label)

                for celebrity in transcript_json["detected_celebrities"]:
                    if float(celebrity["timestamp"]) >= shot_start and float(celebrity["timestamp"]) <= shot_end:
                        div_celebrity = etree.Element('div',
                            type="DetectedPerson"
                        )
                        div_celebrity.attrib["when"] = time.strftime('%H:%M:%S', time.gmtime(float(celebrity["timestamp"])))
                        p = etree.Element('p')
                        persName = etree.Element('persName')
                        persName.text = celebrity["name"]
                        persName.attrib["ref"] = "#"+getReferenceIdByText(celebrity["name"], "celebrity")
                        p.append(persName)
                        div_celebrity.append(p)
                        shot_elements_array.append(div_celebrity)

                
                shot_elements_array.sort(key=lambda x: x.attrib["when"])
                for element in shot_elements_array:
                    div_shot.append(element)

                cue_elements_array.append(div_shot)

        # Utterances

        counter = 1
        for utterance in transcript_json["utterances"]:
            if float(utterance["start"]) >= cue_start and float(utterance["start"]) <= cue_end:
                div_speech = etree.Element('div',
                    type="DetectedSpeech"
                )
                div_speech.attrib["from"] = time.strftime('%H:%M:%S', time.gmtime(float(utterance["start"])))
                div_speech.attrib["to"] = time.strftime('%H:%M:%S', time.gmtime(float(utterance["end"])))
                div_speech.attrib["dur"] = utterance["dur"]

                u = etree.Element('u')
                u.attrib["{http://www.w3.org/XML/1998/namespace}id"] = "utterance"+f'{counter:03d}'
                u.attrib["ana"] = "#"+utterance["sentiment"]
                u.attrib["corresp"] = "#translation"+f'{counter:03d}'
                # Add entities in utterance text
                temp_utterance = utterance["text"]
                
                for entity in utterance["entities"]:
                    if entity["Type"] == "LOCATION":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<placeName ref='#"+getReferenceIdByText(entity["Text"], dictType="place")+"'>"+entity["Text"]+"</placeName>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 27, 39)
                    if entity["Type"] == "PERSON":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<persName ref='#"+getReferenceIdByText(entity["Text"], dictType="pers")+"'>"+entity["Text"]+"</persName>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 25, 36)
                    if entity["Type"] == "DATE":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<date>"+entity["Text"]+"</date>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 6, 13)
                    if entity["Type"] == "ORGANIZATION":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<orgName ref='#"+getReferenceIdByText(entity["Text"], dictType="org")+"'>"+entity["Text"]+"</orgName>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 23, 33)
                    if entity["Type"] == "QUANTITY":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<num>"+entity["Text"]+"</num>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 5, 11)
                
                
                for entity in utterance["syntax"]:
                    w = etree.Element('w')
                    if entity["PartOfSpeech"]["Tag"] == "NOUN":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='NN'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 12, 16)
                    elif entity["PartOfSpeech"]["Tag"] == "DET":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='ART'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 13, 17)
                    elif entity["PartOfSpeech"]["Tag"] == "NUM":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='CARD'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 14, 18)
                    elif entity["PartOfSpeech"]["Tag"] == "ADJ":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='ADJA'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 14, 18)
                    elif entity["PartOfSpeech"]["Tag"] == "ADP":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='APPR'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 14, 18)
                    elif entity["PartOfSpeech"]["Tag"] == "ADV":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='ADV'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 13, 17)
                    elif entity["PartOfSpeech"]["Tag"] == "AUX":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='VAINF'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 15, 19)
                    elif entity["PartOfSpeech"]["Tag"] == "CCONJ":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='KON'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 13, 17)
                    elif entity["PartOfSpeech"]["Tag"] == "INTJ":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='ITJ'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 13, 17)
                    elif entity["PartOfSpeech"]["Tag"] == "PRON":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='PPER'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 14, 18)
                    elif entity["PartOfSpeech"]["Tag"] == "PROPN":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='NE'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 12, 16)
                    elif entity["PartOfSpeech"]["Tag"] == "PUNCT":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='XY'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 12, 16)
                    elif entity["PartOfSpeech"]["Tag"] == "SCONJ":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='KOUS'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 14, 18)
                    elif entity["PartOfSpeech"]["Tag"] == "VERB":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='VVINF'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 15, 19)
                    elif entity["PartOfSpeech"]["Tag"] == "PART":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='PTKZU'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 15, 19)
                    elif entity["PartOfSpeech"]["Tag"] == "SYM":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='XY'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 12, 16)
                    elif entity["PartOfSpeech"]["Tag"] == "O":
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='XY'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 12, 16)
                    else:
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + temp_utterance[entity["EndOffset"] + 0:]
                        temp_utterance = temp_utterance[:entity["BeginOffset"]] + "<w pos='XY'>"+entity["Text"]+"</w>" + temp_utterance[entity["BeginOffset"]:]
                        utterance = increaseOffsets(utterance, entity["BeginOffset"], entity["EndOffset"], 12, 16)
                
                
                fragment = etree.fromstring(f"<temp>{temp_utterance}</temp>")
                u.append(fragment)
                div_speech.append(u)
                ab = etree.Element('ab')
                ab.attrib["{http://www.w3.org/XML/1998/namespace}id"] = "translation"+f'{counter:03d}'
                ab.attrib["{http://www.w3.org/XML/1998/namespace}lang"] = transcript_json["Metadata"]["TranslationLanguage"]
                ab.attrib["type"] = "translation"
                ab.text = utterance["translation"]
                div_speech.append(ab)
                cue_elements_array.append(div_speech)
                etree.strip_tags(div_speech, "temp")
                counter = counter + 1
        
        cue_elements_array.sort(key=lambda x: x.attrib["from"])
        for element in cue_elements_array:
            div_cue.append(element)

        body.append(div_cue)
    
    text.append(body)
    TEI.append(text)

    return TEI

    


def main():
    if startAnnotationJobs():
        #return True
        tei_document = createTeiFile()
        etree.ElementTree(tei_document).write(file_path+tei_output_file_name,
                                            encoding="utf-8",
                                            xml_declaration=True,
                                            pretty_print=True
                                        )



if __name__ == '__main__':
    main()


