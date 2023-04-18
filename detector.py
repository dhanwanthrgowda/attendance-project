from datetime import datetime
import os, signal, logging, time
import cv2
from mtcnn.mtcnn import MTCNN
from embeddings import  save_person_image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from database import  face_from_unkown_people_table, connect_to_database, retrieve_face_vector_from_db, insert_entry_to_database, save_to_unkown_people_table, save_bounding_box

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('detector.log')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

EVENT_IMAGES_PATH = os.environ.get('EVENT_IMAGES_PATH', 'images/')
KNOWN_PATH = os.environ.get('EVENT_IMAGES_PATH', 'sensor_data/')
UNKNOWN_PATH = os.environ.get('EVENT_IMAGES_PATH', 'unkown/')

CAMERA_SOURCE_PATH = os.environ.get('CAMERA_SOURCE_PATH', "http://camera.buffalotrace.com/mjpg/video.mjpg")
CAMERA_TYPE = os.environ.get('CAMERA_TYPE', "ENTRY")
CAMERA_ID = os.environ.get('CAMERA_ID', "Temple_Entance_1")
continue_processing = True

def signal_handler():
    """
    This function sets the process flag to false to end the program
    """
    global continue_processing
    continue_processing = False


signal.signal(signal.SIGTSTP, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def  check_similarity_with_database(database_face_vectors, faceid):
    """
    This functions calculates the face similarity and returns person_id or uid
    """
    for key, value in database_face_vectors.items():
        if is_match(value, faceid[0]):
            return key
    return None

def is_match(known_embedding, candidate_embedding, thresh=0.5):
    """
    Checks similarity between faces based on the cosine distance.
    """
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        return True
    else:
        return False
 
def handle_unkown_person_detection(faceid, faces, face_crop_img, person_id_in_current_frame, coordinates, conn, frame):
    """
    This function checks the person match with any on of the unkown_people table from database
    Saves the new unkown_person in unkown_people table
    """
    results = face_from_unkown_people_table(conn)
    uid = check_similarity_with_database(results, faceid)
    current_time = datetime.now().replace(tzinfo=None)

    if not uid:
        face_vector =  list(faceid[0].astype(float))
        uid = save_to_unkown_people_table(face_vector,current_time, conn)
        image_path = f"{EVENT_IMAGES_PATH}/{UNKNOWN_PATH}/{uid}"
        faces[uid] = {'start': current_time, 'face_crop': face_crop_img,
                'image_path': image_path, 'bounding_box': coordinates, 'frame':frame
                                                                                                                    }
        save_person_image(image_path, face_crop_img)
        save_bounding_box(uid, list(coordinates.values()), current_time, conn)
        
        logger.info("Saved image and bounding_boxes ...")
    if uid not in faces:
        image_path = f"{EVENT_IMAGES_PATH}/{UNKNOWN_PATH}/{uid}/"
        faces[uid]= {'start': current_time, 'face_crop': face_crop_img,
                'image_path': image_path, 'bounding_box': coordinates, "frame":frame
                                                                                                                            }
        save_person_image(image_path, face_crop_img)
        save_bounding_box(uid, list(coordinates.values()), current_time, conn)
        logger.info("Saved image and bounding_boxes ...")
    if uid not in person_id_in_current_frame:
        person_id_in_current_frame.append(uid)
    return person_id_in_current_frame, faces

def get_face_crop(face, face_cooridantes, frame):
    x1, y1, width, height = face
    x2, y2 = x1 + width, y1 + height
    face_cooridantes.append({   'x1':x1,
                                'x2':x2,
                                'y1':y1,
                                'y2':y2
                            })
    face = frame[y1:y2, x1:x2]
    return face, face_cooridantes

def handle_known_person_detection(
                                person_id_in_current_frame, person_id, 
                                faces, coordinates, face_crop_img, frame
                                ):
    """
    This function saves the person image and the frame at first detection.
    """
    person_id_in_current_frame.append(person_id)
    if person_id not in list(faces.keys()):
        current_time = datetime.now()
        image_path = f"{EVENT_IMAGES_PATH}/{KNOWN_PATH}/{person_id}/"
        faces[person_id]= {'start': current_time, 'face_crop': face_crop_img,
                                    'image_path': image_path, 'bounding_box': coordinates, 'frame':frame
                                                                }
        save_person_image(image_path, face_crop_img)
        save_person_image(image_path, frame, flag=1)
    return person_id_in_current_frame, faces

def exit_gracefully(video_stream):
    """
    This function closes all database connections and closes the video stream or camera.
    """
    logger.info("Releasing the Camera Capture...")
    video_stream.release()
    # close_database()
    exit()


def retry_to_open_camera():
    number_of_retry = 0
    video_stream = open_camera(CAMERA_SOURCE_PATH)
    ret, frame = video_stream.read()
    while not ret:
        time.sleep(5)
        number_of_retry += 1
        video_stream = open_camera(CAMERA_SOURCE_PATH)
        ret, frame = video_stream.read()
        if number_of_retry == 5:
            exit_gracefully(video_stream)
    return frame


def match_detected_face_from_database(
                                    frame, face_crop_img, faceid,
                                     person_id_in_current_frame, faces, 
                                     database_face_vectors, coordinates, conn
                                     ):
    """
    Check if face matches with the database else add the unkown person to unkwon people table
    """
    person_id = check_similarity_with_database(database_face_vectors, faceid)
    if person_id == None:
        person_id_in_current_frame, faces = handle_unkown_person_detection(
                                                faceid, faces, face_crop_img,
                                                person_id_in_current_frame, coordinates, conn, frame
                                            )
    else:
        person_id_in_current_frame, faces = handle_known_person_detection(
                                                                        person_id_in_current_frame,
                                                                        person_id, faces,
                                                                        coordinates, face_crop_img, frame
                                                                         )
    cv2.rectangle(frame,(coordinates['x1'],coordinates['y1']),(coordinates['x2'],coordinates['y2']),(0,255,0),2)
    return person_id_in_current_frame, faces


def handle_person_exit_from_camera(  person_id_in_current_frame, faces, conn):
    for key in list(faces.keys()):
        if key not in person_id_in_current_frame:
            current_time = datetime.now()
            total_time_detected = (current_time - faces[key]['start']).total_seconds()
            if total_time_detected > 3:
                print(faces[key])
                image_path = faces[key]['image_path']
                face_crop_img = faces[key]['face_crop']
                event_time = faces[key]['start'].replace(tzinfo=None)
                if isinstance(key, int):
                    insert_entry_to_database(event_time, key, CAMERA_TYPE, CAMERA_ID, conn)
                    save_person_image(image_path, face_crop_img)
                    logger.info("An event is written to sensor_data table ...")
                else:
                    save_person_image(image_path, face_crop_img)
                    save_bounding_box(key, list(faces[key]['bounding_box'].values()), current_time, conn)
                    insert_entry_to_database(event_time,key,  CAMERA_TYPE, CAMERA_ID, conn)
                    logger.info("An event is written to sensor_data table ...")
                save_person_image(image_path, faces[key]['frame'], flag=1)
                    
            del faces[key]
                

def extract_face_coordinates(
                            face, face_crooped_frame, face_cooridantes, 
                            all_faces, frame, required_size=(224, 224)
                            ):
    """
    This function converts the face in image to an array representation.
    """
    face, face_cooridantes = get_face_crop(face, face_cooridantes, frame)
    face_crooped_frame.append(face)
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    all_faces.append(face_array)
    return all_faces, face_crooped_frame, face_cooridantes


def extract_face(detector, frame):
    all_faces = []
    face_cropped_frame = []
    face_cooridantes = []
    results = detector.detect_faces(frame)
    return_extract_face = ([], [], [])
    for i in range(len(results)):
        return_extract_face = extract_face_coordinates(
                                                        results[i]['box'], 
                                                        face_cropped_frame, face_cooridantes, 
                                                        all_faces, frame
                                                        )
    return return_extract_face[0], return_extract_face[1], return_extract_face[2]

def get_embeddings(detector, model, frame):
    all_embeddings = []
    faces, face_crop_frames, face_coordinates  = extract_face(detector, frame)
    
    for i in faces:
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        yhat = model.predict(samples)
        all_embeddings.append(yhat)
    return all_embeddings, face_crop_frames, face_coordinates


def open_camera(source):
    video_stream = cv2.VideoCapture(source)
    if video_stream:
        logger.info(f"Video Camera opened successfully{video_stream}")
    return video_stream
 

def main(video_stream, conn):
    logger.info(f"About to enter the while loop...")
    faces = {}
    database_face_vectors = retrieve_face_vector_from_db(conn)
    detector = MTCNN()
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    frame_num = 1
    total = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total number of frames is :", total )
    while continue_processing:
        person_id_in_current_frame = []
        ret, frame = video_stream.read()

        if ret is False:
            frame = retry_to_open_camera()
        
        all_embedds, face_cropped_frame, face_coordinates = get_embeddings(detector,model, frame)
        for index in range(len(all_embedds)):
            match_detected_face_from_database_params = {
                'face_crop_img': face_cropped_frame[index],
                 'faceid':all_embedds[index], 
                 'person_id_in_current_frame':person_id_in_current_frame, 
                 'faces':faces, 
                 'database_face_vectors': database_face_vectors, 
                 'conn': conn,
                 'coordinates': face_coordinates[index],
                 'frame':frame
            }
            person_id_in_current_frame, faces = match_detected_face_from_database(**match_detected_face_from_database_params)
        handle_person_exit_from_camera(person_id_in_current_frame, faces, conn)
        frame_num+=1
        print(f"Frame number is : {frame_num}/{total}")
        if total == frame_num:
            break

if __name__ == '__main__':
    try:
        logger.info(f"Opening the Video Feed...")
        video_stream = open_camera(CAMERA_SOURCE_PATH)
        conn = connect_to_database()
        logger.info(f"Connected to the Database Successfully")
        main(video_stream, conn)
    except cv2.error as error:
        logger.error(f"Cannot open the video {error}")
    except Exception as error:
        logger.info(f"Unexpected Error {error}")
    finally :
        exit()
