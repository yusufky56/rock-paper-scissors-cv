import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import random
from collections import deque
import statistics
import numpy as np
import math

from rps_logic import MOVES, classify_gesture, round_winner

class UltimateRockPaperScissorsDetector:
    def __init__(self):
        # MediaPipe el algılama kütüphaneleri
        self.mediapipe_hands = mp.solutions.hands
        self.hand_drawer = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        
        # El landmark noktalarının indeks haritası
        self.hand_landmarks_map = {
            'wrist': 0,
            'thumb_cmc': 1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4,
            'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
            'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
            'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
            'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20
        }
        
        # Hareket geçmişi ve oyun durumu
        self.gesture_buffer = deque(['none'] * 5, maxlen=5)
        self.current_gesture = 'none'
        self.is_game_active = False
        self.has_round_finished = False
        self.was_hand_detected_before = False
        
        # Skor takibi
        self.player_total_score = 0
        self.computer_total_score = 0
        self.computer_current_move = 'none'
        self.match_result = ''
        self.result_display_color = (255, 255, 255)
        
        # Oyun hareketleri ve ikon önbelleği
        self.available_moves = list(MOVES)
        self.move_icon_cache = {}
        self.initialize_move_icons()
    
    def initialize_move_icons(self):
        # Hareket ikonlarını önceden oluştur
        for gesture in ['rock', 'paper', 'scissors', 'none']:
            self.move_icon_cache[gesture] = self.create_gesture_icon(gesture)

    def create_gesture_icon(self, gesture_type, icon_size=100):
        # Hareket tipine göre ikon oluştur
        icon_canvas = np.zeros((icon_size, icon_size, 3), dtype=np.uint8)
        icon_center = (icon_size // 2, icon_size // 2)
        main_color = (80, 200, 80)
        font_color = (255, 255, 255)
        
        #iconların çizilmesi
        if gesture_type == 'rock':
            cv2.circle(icon_canvas, icon_center, icon_size // 3, main_color, -1)
            cv2.putText(icon_canvas, 'R', (icon_center[0] - 12, icon_center[1] + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        elif gesture_type == 'paper':
            cv2.rectangle(icon_canvas, (icon_size // 4, icon_size // 4), 
                         (3 * icon_size // 4, 3 * icon_size // 4), main_color, -1)
            cv2.putText(icon_canvas, 'P', (icon_center[0] - 12, icon_center[1] + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        elif gesture_type == 'scissors':
            cv2.line(icon_canvas, (icon_size // 4, icon_size // 4), 
                    (3 * icon_size // 4, 3 * icon_size // 4), main_color, 5)
            cv2.line(icon_canvas, (icon_size // 4, 3 * icon_size // 4), 
                    (3 * icon_size // 4, icon_size // 4), main_color, 5)
            cv2.putText(icon_canvas, 'S', (icon_center[0] - 12, icon_center[1] + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        return icon_canvas
    
    def calculate_point_distance(self, point_a, point_b):
        # İki nokta arasındaki Öklid mesafesini hesapla
        return math.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)
    
    def check_if_finger_extended(self, hand_landmarks, tip_index, pip_index, mcp_index):
        # Parmağın uzatılıp uzatılmadığını kontrol et
        tip_point = hand_landmarks[tip_index]
        pip_point = hand_landmarks[pip_index]
        
        # Parmak ucu, orta boğumdan yukarıdaysa uzatılmış
        return tip_point.y < pip_point.y
    
    def detect_thumb_extension(self, hand_landmarks, hand_orientation):
        # Başparmağın uzatılıp uzatılmadığını tespit et
        thumb_tip = hand_landmarks[self.hand_landmarks_map['thumb_tip']]
        thumb_ip = hand_landmarks[self.hand_landmarks_map['thumb_ip']]
        thumb_mcp = hand_landmarks[self.hand_landmarks_map['thumb_mcp']]
        
        # Başparmak ucu, MCP'den IP'den daha uzaktaysa uzatılmış
        tip_to_mcp = self.calculate_point_distance(
            [thumb_tip.x, thumb_tip.y], [thumb_mcp.x, thumb_mcp.y]
        )
        ip_to_mcp = self.calculate_point_distance(
            [thumb_ip.x, thumb_ip.y], [thumb_mcp.x, thumb_mcp.y]
        )
        
        return tip_to_mcp > ip_to_mcp * 1.3
    
    def analyze_hand_gesture(self, hand_landmarks, hand_orientation):
        # El hareketini analiz et ve taş-kağıt-makas belirle
        finger_tip_indices = [
            self.hand_landmarks_map['index_tip'],
            self.hand_landmarks_map['middle_tip'],
            self.hand_landmarks_map['ring_tip'],
            self.hand_landmarks_map['pinky_tip']
        ]
        
        finger_pip_indices = [
            self.hand_landmarks_map['index_pip'],
            self.hand_landmarks_map['middle_pip'],
            self.hand_landmarks_map['ring_pip'],
            self.hand_landmarks_map['pinky_pip']
        ]
        
        finger_mcp_indices = [
            self.hand_landmarks_map['index_mcp'],
            self.hand_landmarks_map['middle_mcp'],
            self.hand_landmarks_map['ring_mcp'],
            self.hand_landmarks_map['pinky_mcp']
        ]
        
        # 4 parmağın durumunu kontrol et (başparmak hariç)
        extended_fingers_count = 0
        finger_states = []
        
        for i in range(4):
            is_extended = self.check_if_finger_extended(
                hand_landmarks, finger_tip_indices[i], 
                finger_pip_indices[i], finger_mcp_indices[i]
            )
            finger_states.append(is_extended)
            if is_extended:
                extended_fingers_count += 1
        
        # Başparmağı ayrı kontrol et
        thumb_extended = self.detect_thumb_extension(hand_landmarks, hand_orientation)
        
        # Hareket sınıflandırması (kurallar rps_logic.py içinde)
        return classify_gesture(finger_states)

    def update_gesture_history(self, detected_gesture):
        # Hareket geçmişini güncelle ve en yaygın hareketi belirle
        self.gesture_buffer.appendleft(detected_gesture)
        
        try:
            most_common_gesture = statistics.mode(self.gesture_buffer)
            if most_common_gesture in self.available_moves:
                self.current_gesture = most_common_gesture
            else:
                self.current_gesture = 'invalid'
        except statistics.StatisticsError:
            if detected_gesture in self.available_moves:
                self.current_gesture = detected_gesture
            else:
                self.current_gesture = 'invalid'
    
    def handle_game_state(self, is_hand_currently_detected):
        # Oyun durumunu yönet
        
        # El görüldü - oyunu başlat
        if is_hand_currently_detected and not self.was_hand_detected_before:
            self.is_game_active = True
            self.has_round_finished = False
            self.gesture_buffer.clear()
            for _ in range(self.gesture_buffer.maxlen):
                self.gesture_buffer.appendleft('none')
        
        # El kayboldu - oyunu sıfırla
        elif not is_hand_currently_detected and self.was_hand_detected_before:
            self.reset_current_game()
        
        # Oyun turu oyna
        if (self.current_gesture in self.available_moves and 
            not self.has_round_finished and 
            self.is_game_active and 
            is_hand_currently_detected):
            self.execute_game_round()
        
        self.was_hand_detected_before = is_hand_currently_detected
    
    def execute_game_round(self):
        # Oyun turunu gerçekleştir
        self.computer_current_move = random.choice(self.available_moves)
        self.match_result = self.calculate_round_winner(self.current_gesture, self.computer_current_move)
        self.has_round_finished = True
        
        # Skoru güncelle ve renk ayarla
        if self.match_result == 'player_victory':
            self.player_total_score += 1
            self.result_display_color = (0, 255, 0)  # Yeşil
        elif self.match_result == 'computer_victory':
            self.computer_total_score += 1
            self.result_display_color = (0, 0, 255)  # Kırmızı
        elif self.match_result == 'draw':
            self.result_display_color = (255, 255, 0)  # Sarı
    
    def calculate_round_winner(self, player_move, computer_move):
        # Tur kazananını hesapla
        return round_winner(player_move, computer_move)

    def reset_current_game(self):
        # Mevcut oyunu sıfırla
        self.is_game_active = False
        self.has_round_finished = False
        self.computer_current_move = 'none'
        self.match_result = ''
        self.current_gesture = 'none'
    
    def render_game_interface(self, video_frame):
        # Oyun arayüzünü çiz
        frame_height, frame_width = video_frame.shape[:2]
        display_font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Oyuncu bölümü
        cv2.putText(video_frame, "OYUNCU", (30, 40), display_font, 1, (255, 120, 120), 2)
        cv2.putText(video_frame, f"Skor: {self.player_total_score}", (30, 75), 
                   display_font, 0.8, (255, 120, 120), 2)
        
        # Mevcut hareket gösterimi
        gesture_text = self.current_gesture.upper() if self.current_gesture != 'none' else 'BEKLIYOR'
        if self.current_gesture == 'invalid':
            gesture_text = 'GECERSIZ'
        elif not self.is_game_active:
            gesture_text = 'YOK'
        
        cv2.putText(video_frame, f"Hareket: {gesture_text}", (30, 110), 
                   display_font, 0.7, (255, 120, 120), 2)
        
        # Bilgisayar bölümü
        computer_section_x = frame_width - 180
        cv2.putText(video_frame, "BILGISAYAR", (computer_section_x, 40), 
                   display_font, 1, (120, 120, 255), 2)
        cv2.putText(video_frame, f"Skor: {self.computer_total_score}", 
                   (computer_section_x, 75), display_font, 0.8, (120, 120, 255), 2)
        
        # Bilgisayar hareket ikonu
        if self.computer_current_move != 'none':
            computer_icon = self.move_icon_cache[self.computer_current_move]
            icon_dimension = computer_icon.shape[0]
            icon_y_pos, icon_x_pos = 20, frame_width - icon_dimension - 20
            
            if icon_y_pos + icon_dimension <= frame_height and icon_x_pos >= 0:
                try:
                    # İkonu video frame'ine yerleştir
                    icon_region = video_frame[icon_y_pos:icon_y_pos + icon_dimension, 
                                            icon_x_pos:icon_x_pos + icon_dimension]
                    icon_grayscale = cv2.cvtColor(computer_icon, cv2.COLOR_BGR2GRAY)
                    _, icon_mask = cv2.threshold(icon_grayscale, 1, 255, cv2.THRESH_BINARY)
                    inverted_mask = cv2.bitwise_not(icon_mask)
                    
                    background_part = cv2.bitwise_and(icon_region, icon_region, mask=inverted_mask)
                    foreground_part = cv2.bitwise_and(computer_icon, computer_icon, mask=icon_mask)
                    final_icon = cv2.add(background_part, foreground_part)
                    
                    video_frame[icon_y_pos:icon_y_pos + icon_dimension, 
                              icon_x_pos:icon_x_pos + icon_dimension] = final_icon
                except:
                    pass
        
        # Oyun sonucu gösterimi
        if self.match_result:
            result_message_map = {
                'player_victory': 'KAZANDINIZ!',
                'computer_victory': 'BILGISAYAR KAZANDI!',
                'draw': 'BERABERE!'
            }
            
            result_message = result_message_map.get(self.match_result, '')
            if result_message:
                text_dimensions = cv2.getTextSize(result_message, display_font, 1.2, 3)[0]
                message_x_position = (frame_width - text_dimensions[0]) // 2
                cv2.putText(video_frame, result_message, (message_x_position, frame_height - 50), 
                           display_font, 1.2, self.result_display_color, 3)
        
        # El algılanmadığında talimatlar
        if not self.is_game_active:
            cv2.putText(video_frame, "Elinizi kameraya gosterin!", 
                       (frame_width // 2 - 140, frame_height // 2), 
                       display_font, 0.9, (200, 200, 200), 2)
        
        return video_frame

def launch_game():
    # Ana oyun fonksiyonu
    try:
        # Kamera başlatma
        camera_capture = cv2.VideoCapture(0)
        camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera_capture.set(cv2.CAP_PROP_FPS, 60)
        
        if not camera_capture.isOpened():
            print("Hata: Kameraya erisim saglanamadi!")
            return
        
        # Oyun dedektörü başlat
        game_detector = UltimateRockPaperScissorsDetector()
        
        # MediaPipe el algılama yapılandırması
        with game_detector.mediapipe_hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as hand_processor:
            
            print("Mukemmel Tas-Kagit-Makas Oyunu Baslatildi!")
            print("Talimatlar: Kameraya tas, kagit veya makas gosterin")
            print("Cikmak icin ESC tusuna basin")
            
            frame_counter = 0
            
            # Ana oyun döngüsü
            while True:
                capture_successful, current_frame = camera_capture.read()
                if not capture_successful:
                    break
                
                # Performans için her 2. frame'i işle
                frame_counter += 1
                if frame_counter % 2 != 0:
                    continue
                
                # Frame'i aynala ve RGB'ye çevir
                current_frame = cv2.flip(current_frame, 1)
                rgb_converted_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                rgb_converted_frame.flags.writeable = False
                
                # El algılama işlemi
                hand_detection_results = hand_processor.process(rgb_converted_frame)
                
                # Frame'i tekrar BGR'ye çevir
                rgb_converted_frame.flags.writeable = True
                current_frame = cv2.cvtColor(rgb_converted_frame, cv2.COLOR_RGB2BGR)
                
                detected_gesture = 'none'
                is_hand_present = False
                
                # El algılandıysa işle
                if hand_detection_results.multi_hand_landmarks:
                    is_hand_present = True
                    primary_hand_landmarks = hand_detection_results.multi_hand_landmarks[0]
                    hand_classification = hand_detection_results.multi_handedness[0]
                    hand_side = hand_classification.classification[0].label
                    
                    # El landmark'larını çiz
                    game_detector.hand_drawer.draw_landmarks(
                        current_frame, primary_hand_landmarks, 
                        game_detector.mediapipe_hands.HAND_CONNECTIONS,
                        game_detector.drawing_styles.get_default_hand_landmarks_style(),
                        game_detector.drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Hareket analizi yap
                    detected_gesture = game_detector.analyze_hand_gesture(
                        primary_hand_landmarks.landmark, hand_side
                    )
                
                # Hareket geçmişini güncelle ve oyunu yönet
                game_detector.update_gesture_history(detected_gesture)
                game_detector.handle_game_state(is_hand_present)
                current_frame = game_detector.render_game_interface(current_frame)
                
                # Frame'i göster
                cv2.imshow('Mukemmel Tas Kagit Makas Oyunu', current_frame)
                
                # ESC tuşuyla çıkış
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        
        # Temizlik
        camera_capture.release()
        cv2.destroyAllWindows()
        print("Oyun basariyla sonlandirildi!")
        
    except Exception as error:
        print(f"Bir hata olustu: {error}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    launch_game()