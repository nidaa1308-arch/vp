CREATE DATABASE IF NOT EXISTS parkvision;
USE parkvision;
CREATE TABLE IF NOT EXISTS users(
 id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(100) NOT NULL,
 email VARCHAR(190) NOT NULL UNIQUE, phone VARCHAR(20),
 password_hash VARCHAR(100) NOT NULL,
 role ENUM('user','admin') NOT NULL DEFAULT 'user',
 status ENUM('active','suspended') NOT NULL DEFAULT 'active',
 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS vehicles(
 id INT AUTO_INCREMENT PRIMARY KEY, user_id INT NOT NULL,
 registration_number VARCHAR(20) NOT NULL UNIQUE,
 vehicle_type ENUM('car','suv','bike') NOT NULL DEFAULT 'car',
 make VARCHAR(50), model VARCHAR(50), colour VARCHAR(30), is_ev BOOLEAN DEFAULT FALSE,
 FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS parking_slots(
 id INT AUTO_INCREMENT PRIMARY KEY, slot_code VARCHAR(10) NOT NULL UNIQUE,
 zone VARCHAR(10) NOT NULL, floor INT NOT NULL DEFAULT 1,
 slot_type ENUM('regular','ev','accessible','reserved') NOT NULL DEFAULT 'regular',
 operational_status ENUM('active','maintenance','blocked') NOT NULL DEFAULT 'active',
 current_occupancy ENUM('free','occupied','unknown') NOT NULL DEFAULT 'unknown',
 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS reservations(
 id INT AUTO_INCREMENT PRIMARY KEY, booking_reference VARCHAR(20) NOT NULL UNIQUE,
 user_id INT NOT NULL, vehicle_id INT NOT NULL, parking_slot_id INT NOT NULL,
 start_time DATETIME NOT NULL, end_time DATETIME NOT NULL,
 reservation_status ENUM('confirmed','completed','cancelled','no_show') NOT NULL DEFAULT 'confirmed',
 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
 INDEX(parking_slot_id,start_time,end_time),
 FOREIGN KEY(user_id) REFERENCES users(id), FOREIGN KEY(vehicle_id) REFERENCES vehicles(id),
 FOREIGN KEY(parking_slot_id) REFERENCES parking_slots(id));
CREATE TABLE IF NOT EXISTS occupancy_events(
 id BIGINT AUTO_INCREMENT PRIMARY KEY, parking_slot_id INT NOT NULL,
 source ENUM('camera','sensor') NOT NULL, observed_status ENUM('free','occupied') NOT NULL,
 confidence DECIMAL(4,3), observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
 INDEX(parking_slot_id,observed_at), FOREIGN KEY(parking_slot_id) REFERENCES parking_slots(id));
CREATE TABLE IF NOT EXISTS audit_logs(
 id BIGINT AUTO_INCREMENT PRIMARY KEY, actor_user_id INT, action VARCHAR(60),
 entity_type VARCHAR(30), entity_id INT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
-- TEST DATA (not real sensor data): 12 slots, occupancy left 'unknown' until hardware reports
INSERT IGNORE INTO parking_slots(slot_code,zone,floor,slot_type) VALUES
('A1','A',1,'regular'),('A2','A',1,'regular'),('A3','A',1,'regular'),('A4','A',1,'accessible'),
('B1','B',1,'regular'),('B2','B',1,'regular'),('B3','B',1,'reserved'),('B4','B',1,'regular'),
('EV1','C',1,'ev'),('EV2','C',1,'ev'),('EV3','C',1,'ev'),('EV4','C',1,'ev');
