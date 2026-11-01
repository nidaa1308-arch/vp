
require('dotenv').config();

const express = require('express');
const http = require('http');
const path = require('path');
const crypto = require('crypto');
const mysql = require('mysql2/promise');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const cookieParser = require('cookie-parser');
const rateLimit = require('express-rate-limit');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const prod = process.env.NODE_ENV === 'production';
const demoMode = !prod && process.env.DEMO_MODE === 'true';

if (prod && !process.env.JWT_SECRET) {
  throw new Error('JWT_SECRET must be configured in production');
}

const jwtSecret = process.env.JWT_SECRET || 'local-demo-secret-change-later';

const pool = mysql.createPool({
  host: process.env.DB_HOST || 'localhost',
  port: Number(process.env.DB_PORT || 3306),
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'parkvision',
  timezone: 'Z',
  connectionLimit: 10,
  waitForConnections: true
});

app.use(express.json({ limit: '10kb' }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, '..')));

const wrap = fn => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch(next);

const bad = (res, message, code = 400) =>
  res.status(code).json({ ok: false, error: message });

const ok = (res, data = {}) =>
  res.json({ ok: true, ...data });

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 30
});

const DEMO_USER = {
  id: 1,
  name: 'ParkVision Demo',
  email: 'demo@parkvision.com',
  role: 'admin',
  status: 'active',
  demo: true
};

const DEMO_SLOTS = [
  { id: 1, slot_code: 'A1', zone: 'A', floor: 1, slot_type: 'regular', operational_status: 'active', current_occupancy: 'free', status: 'available' },
  { id: 2, slot_code: 'A2', zone: 'A', floor: 1, slot_type: 'regular', operational_status: 'active', current_occupancy: 'occupied', status: 'occupied' },
  { id: 3, slot_code: 'A3', zone: 'A', floor: 1, slot_type: 'accessible', operational_status: 'active', current_occupancy: 'free', status: 'available' },
  { id: 4, slot_code: 'B1', zone: 'B', floor: 1, slot_type: 'regular', operational_status: 'active', current_occupancy: 'free', status: 'available' },
  { id: 5, slot_code: 'B2', zone: 'B', floor: 1, slot_type: 'regular', operational_status: 'active', current_occupancy: 'occupied', status: 'occupied' },
  { id: 6, slot_code: 'B3', zone: 'B', floor: 1, slot_type: 'reserved', operational_status: 'active', current_occupancy: 'free', status: 'reserved' },
  { id: 7, slot_code: 'EV1', zone: 'C', floor: 1, slot_type: 'ev', operational_status: 'active', current_occupancy: 'free', status: 'available' },
  { id: 8, slot_code: 'EV2', zone: 'C', floor: 1, slot_type: 'ev', operational_status: 'active', current_occupancy: 'occupied', status: 'occupied' }
];

const DEMO_VEHICLES = [
  { id: 1, registration_number: 'MH12AB1234', vehicle_type: 'car', make: 'Demo', model: 'Car', colour: 'Blue', is_ev: false },
  { id: 2, registration_number: 'MH12EV2026', vehicle_type: 'car', make: 'Demo', model: 'EV', colour: 'Green', is_ev: true }
];

const DEMO_RESERVATIONS = [];

const setCookie = (res, id, demo = false) => {
  res.cookie(
    'pv_token',
    jwt.sign({ id, demo }, jwtSecret, { expiresIn: '8h' }),
    {
      httpOnly: true,
      sameSite: 'strict',
      secure: prod,
      maxAge: 8 * 60 * 60 * 1000
    }
  );
};

const auth = wrap(async (req, res, next) => {
  try {
    const payload = jwt.verify(req.cookies.pv_token || '', jwtSecret);

    if (demoMode && payload.demo === true && payload.id === DEMO_USER.id) {
      req.user = { ...DEMO_USER };
      return next();
    }

    const [[user]] = await pool.execute(
      'SELECT id,name,email,role,status FROM users WHERE id=?',
      [payload.id]
    );

    if (!user || user.status !== 'active') {
      return bad(res, 'Account unavailable', 401);
    }

    req.user = user;
    return next();
  } catch {
    return bad(res, 'Please log in', 401);
  }
});

const admin = (req, res, next) =>
  req.user.role === 'admin' ? next() : bad(res, 'Admin only', 403);

const changed = () => io.emit('slots:changed');

const SLOT_SQL = `
SELECT s.id,s.slot_code,s.zone,s.floor,s.slot_type,
s.operational_status,s.current_occupancy,s.updated_at,
CASE
  WHEN s.operational_status <> 'active' THEN 'unavailable'
  WHEN s.current_occupancy = 'occupied' THEN 'occupied'
  WHEN EXISTS (
    SELECT 1 FROM reservations r
    WHERE r.parking_slot_id=s.id
      AND r.reservation_status='confirmed'
      AND r.start_time<=UTC_TIMESTAMP()
      AND r.end_time>UTC_TIMESTAMP()
  ) THEN 'reserved'
  ELSE 'available'
END AS status
FROM parking_slots s
`;

// Health check
app.get('/api/health', (req, res) =>
  ok(res, { service: 'ParkVision API', demoMode })
);

// Authentication
app.post('/api/auth/login', limiter, wrap(async (req, res) => {
  const { email, password } = req.body || {};

  if (demoMode) {
    if (email !== 'demo@parkvision.com' || password !== 'ParkVision123') {
      return bad(res, 'Use the demo login credentials.', 401);
    }

    setCookie(res, DEMO_USER.id, true);
    return ok(res, { user: DEMO_USER, role: 'admin', demo: true });
  }

  const [[user]] = await pool.execute(
    'SELECT * FROM users WHERE email=?',
    [email || '']
  );

  if (!user || !(await bcrypt.compare(password || '', user.password_hash))) {
    return bad(res, 'Invalid email or password', 401);
  }

  if (user.status !== 'active') {
    return bad(res, 'Account suspended', 403);
  }

  setCookie(res, user.id);
  return ok(res, {
    user: { id: user.id, name: user.name, email: user.email, role: user.role },
    role: user.role
  });
}));

app.post('/api/auth/register', limiter, wrap(async (req, res) => {
  if (demoMode) {
    return bad(res, 'Registration is disabled in demo mode.', 403);
  }

  const { name, email, password, phone } = req.body || {};

  if (
    !name ||
    name.length > 100 ||
    !/^\S+@\S+\.\S+$/.test(email || '') ||
    (password || '').length < 8
  ) {
    return bad(res, 'Enter a valid name, email and password of at least 8 characters.');
  }

  const [[existing]] = await pool.execute(
    'SELECT id FROM users WHERE email=?',
    [email]
  );

  if (existing) return bad(res, 'Email already registered', 409);

  const hash = await bcrypt.hash(password, 12);

  const [result] = await pool.execute(
    'INSERT INTO users(name,email,phone,password_hash) VALUES(?,?,?,?)',
    [name, email, phone || null, hash]
  );

  setCookie(res, result.insertId);
  return ok(res, { role: 'user' });
}));

app.post('/api/auth/logout', (req, res) => {
  res.clearCookie('pv_token', {
    httpOnly: true,
    sameSite: 'strict',
    secure: prod
  });
  return ok(res);
});

app.get('/api/auth/me', auth, (req, res) =>
  ok(res, { user: req.user, demo: !!req.user.demo })
);

// Parking summary
app.get('/api/parking/summary', auth, wrap(async (req, res) => {
  if (req.user.demo && demoMode) {
    const count = status =>
      DEMO_SLOTS.filter(slot => slot.status === status).length;

    const evSlots = DEMO_SLOTS.filter(slot => slot.slot_type === 'ev');

    return ok(res, {
      total: DEMO_SLOTS.length,
      available: count('available'),
      occupied: count('occupied'),
      reserved: count('reserved'),
      unavailable: count('unavailable'),
      activeReservations: 0,
      evAvailable: evSlots.filter(slot => slot.status === 'available').length,
      evTotal: evSlots.length,
      mine: 0,
      demo: true
    });
  }

  const [slots] = await pool.query(SLOT_SQL);
  const count = status => slots.filter(slot => slot.status === status).length;

  const [[mine]] = await pool.execute(
    "SELECT COUNT(*) n FROM reservations WHERE user_id=? AND reservation_status='confirmed' AND end_time>UTC_TIMESTAMP()",
    [req.user.id]
  );

  const [[all]] = await pool.query(
    "SELECT COUNT(*) n FROM reservations WHERE reservation_status='confirmed' AND end_time>UTC_TIMESTAMP()"
  );

  const evSlots = slots.filter(slot => slot.slot_type === 'ev');

  return ok(res, {
    total: slots.length,
    available: count('available'),
    occupied: count('occupied'),
    reserved: count('reserved'),
    unavailable: count('unavailable'),
    activeReservations: all.n,
    evAvailable: evSlots.filter(slot => slot.status === 'available').length,
    evTotal: evSlots.length,
    mine: mine.n
  });
}));

app.get('/api/parking', auth, wrap(async (req, res) => {
  if (req.user.demo && demoMode) {
    return ok(res, { slots: DEMO_SLOTS, demo: true });
  }

  const [slots] = await pool.query(SLOT_SQL + ' ORDER BY s.zone,s.slot_code');
  return ok(res, { slots });
}));

// Vehicles
app.get('/api/vehicles', auth, wrap(async (req, res) => {
  if (req.user.demo && demoMode) {
    return ok(res, { vehicles: DEMO_VEHICLES, demo: true });
  }

  const [vehicles] = await pool.execute(
    'SELECT * FROM vehicles WHERE user_id=?',
    [req.user.id]
  );

  return ok(res, { vehicles });
}));

app.post('/api/vehicles', auth, wrap(async (req, res) => {
  if (req.user.demo && demoMode) {
    return bad(res, 'Vehicle changes are disabled in demo mode.', 403);
  }

  const body = req.body || {};
  const reg = (body.registration_number || '').toUpperCase().trim();

  if (!/^[A-Z0-9 -]{4,20}$/.test(reg)) {
    return bad(res, 'Invalid registration number');
  }

  await pool.execute(
    'INSERT INTO vehicles(user_id,registration_number,vehicle_type,make,model,colour,is_ev) VALUES(?,?,?,?,?,?,?)',
    [
      req.user.id,
      reg,
      ['car', 'suv', 'bike'].includes(body.vehicle_type) ? body.vehicle_type : 'car',
      body.make || null,
      body.model || null,
      body.colour || null,
      !!body.is_ev
    ]
  );

  return ok(res);
}));

// Reservations
app.get('/api/reservations/my', auth, wrap(async (req, res) => {
  if (req.user.demo && demoMode) {
    return ok(res, { reservations: DEMO_RESERVATIONS, demo: true });
  }

  const [reservations] = await pool.execute(
    `SELECT r.id,r.booking_reference,r.start_time,r.end_time,
      r.reservation_status,p.slot_code,v.registration_number
     FROM reservations r
     JOIN parking_slots p ON p.id=r.parking_slot_id
     JOIN vehicles v ON v.id=r.vehicle_id
     WHERE r.user_id=?
     ORDER BY r.start_time DESC LIMIT 100`,
    [req.user.id]
  );

  return ok(res, { reservations });
}));

app.post('/api/reservations', auth, wrap(async (req, res) => {
  if (req.user.demo && demoMode) {
    return bad(res, 'Booking is disabled in demo mode.', 403);
  }

  const { slot_id, vehicle_id, start_time, end_time } = req.body || {};
  const start = new Date(start_time);
  const end = new Date(end_time);

  if (
    isNaN(start) || isNaN(end) ||
    end <= start ||
    end - start > 24 * 60 * 60 * 1000 ||
    start < Date.now() - 60000
  ) {
    return bad(res, 'Invalid time range. Maximum booking is 24 hours.');
  }

  const connection = await pool.getConnection();

  try {
    await connection.beginTransaction();

    const [[slot]] = await connection.execute(
      'SELECT * FROM parking_slots WHERE id=? FOR UPDATE',
      [slot_id]
    );

    if (!slot || slot.operational_status !== 'active') {
      await connection.rollback();
      return bad(res, 'Slot unavailable', 409);
    }

    const [[vehicle]] = await connection.execute(
      'SELECT * FROM vehicles WHERE id=? AND user_id=?',
      [vehicle_id, req.user.id]
    );

    if (!vehicle) {
      await connection.rollback();
      return bad(res, 'Vehicle not found', 404);
    }

    if (slot.slot_type === 'ev' && !vehicle.is_ev) {
      await connection.rollback();
      return bad(res, 'An EV slot requires an eligible vehicle.', 409);
    }

    const [overlaps] = await connection.execute(
      "SELECT id FROM reservations WHERE parking_slot_id=? AND reservation_status='confirmed' AND start_time<? AND end_time>?",
      [slot.id, end, start]
    );

    if (overlaps.length) {
      await connection.rollback();
      return bad(res, 'Slot already booked for that time.', 409);
    }

    const reference = 'PV-' + crypto.randomBytes(4).toString('hex').toUpperCase();

    await connection.execute(
      'INSERT INTO reservations(booking_reference,user_id,vehicle_id,parking_slot_id,start_time,end_time) VALUES(?,?,?,?,?,?)',
      [reference, req.user.id, vehicle.id, slot.id, start, end]
    );

    await connection.commit();
    changed();

    return ok(res, { booking_reference: reference });
  } catch (err) {
    await connection.rollback();
    throw err;
  } finally {
    connection.release();
  }
}));

app.patch('/api/reservations/:id/cancel', auth, wrap(async (req, res) => {
  if (req.user.demo && demoMode) {
    return bad(res, 'Cancellation is disabled in demo mode.', 403);
  }

  const [result] = await pool.execute(
    "UPDATE reservations SET reservation_status='cancelled' WHERE id=? AND user_id=? AND reservation_status='confirmed'",
    [req.params.id, req.user.id]
  );

  if (!result.affectedRows) {
    return bad(res, 'Reservation cannot be cancelled', 404);
  }

  changed();
  return ok(res);
}));

// Admin summary
app.get('/api/admin/summary', auth, admin, wrap(async (req, res) => {
  if (req.user.demo && demoMode) {
    return ok(res, {
      users: 1,
      reservations: DEMO_RESERVATIONS.length,
      lastReading: null,
      demo: true
    });
  }

  const [[users]] = await pool.query('SELECT COUNT(*) n FROM users');
  const [[reservations]] = await pool.query('SELECT COUNT(*) n FROM reservations');
  const [[reading]] = await pool.query(
    'SELECT MAX(observed_at) last_reading FROM occupancy_events'
  );

  return ok(res, {
    users: users.n,
    reservations: reservations.n,
    lastReading: reading.last_reading
  });
}));

app.patch('/api/admin/parking-slots/:id', auth, admin, wrap(async (req, res) => {
  if (req.user.demo && demoMode) {
    return bad(res, 'Editing slots is disabled in demo mode.', 403);
  }

  const { operational_status, slot_type } = req.body || {};

  if (operational_status &&
      !['active', 'maintenance', 'blocked'].includes(operational_status)) {
    return bad(res, 'Invalid operational status');
  }

  if (slot_type &&
      !['regular', 'ev', 'accessible', 'reserved'].includes(slot_type)) {
    return bad(res, 'Invalid slot type');
  }

  await pool.execute(
    'UPDATE parking_slots SET operational_status=COALESCE(?,operational_status),slot_type=COALESCE(?,slot_type) WHERE id=?',
    [operational_status || null, slot_type || null, req.params.id]
  );

  await pool.execute(
    'INSERT INTO audit_logs(actor_user_id,action,entity_type,entity_id) VALUES(?,?,?,?)',
    [req.user.id, 'update_slot', 'slot', req.params.id]
  );

  changed();
  return ok(res);
}));

// IoT / camera occupancy observations
app.post('/api/iot/observation', wrap(async (req, res) => {
  if (demoMode) {
    return bad(res, 'IoT updates require the real database.', 403);
  }

  if (!process.env.DEVICE_API_KEY ||
      req.get('x-device-key') !== process.env.DEVICE_API_KEY) {
    return bad(res, 'Unauthorized', 401);
  }

  const { slot_code, source, status, confidence } = req.body || {};

  if (!['camera', 'sensor'].includes(source) ||
      !['free', 'occupied'].includes(status)) {
    return bad(res, 'Invalid observation');
  }

  const [[slot]] = await pool.execute(
    'SELECT id FROM parking_slots WHERE slot_code=?',
    [slot_code]
  );

  if (!slot) return bad(res, 'Unknown slot', 404);

  await pool.execute(
    'INSERT INTO occupancy_events(parking_slot_id,source,observed_status,confidence) VALUES(?,?,?,?)',
    [slot.id, source, status, confidence ?? null]
  );

  const [[other]] = await pool.execute(
    "SELECT observed_status FROM occupancy_events WHERE parking_slot_id=? AND source<>? AND observed_at>UTC_TIMESTAMP()-INTERVAL 30 SECOND ORDER BY observed_at DESC LIMIT 1",
    [slot.id, source]
  );

  const finalStatus = other && other.observed_status !== status
    ? 'unknown'
    : status;

  await pool.execute(
    'UPDATE parking_slots SET current_occupancy=? WHERE id=?',
    [finalStatus, slot.id]
  );

  changed();
  return ok(res, { final: finalStatus });
}));

// Complete expired reservations
const expiryTimer = setInterval(() => {
  if (demoMode) return;

  pool.query(
    "UPDATE reservations SET reservation_status='completed' WHERE reservation_status='confirmed' AND end_time<UTC_TIMESTAMP()"
  ).then(changed).catch(err => {
    console.error('Reservation expiry:', err.message);
  });
}, 60000);

expiryTimer.unref();

app.use((err, req, res, next) => {
  console.error(err.message);

  if (res.headersSent) return next(err);

  return bad(res, 'Server error. Check the backend terminal.', 500);
});

app.get('/api/health', (req, res) =>
  ok(res, { service: 'ParkVision API', demoMode })
);

server.listen(Number(process.env.PORT || 3000), () => {
  console.log(
    `ParkVision running on port ${process.env.PORT || 3000}` +
    (demoMode ? ' (DEMO MODE)' : '')
  );

  if (demoMode) {
    console.log('Demo login: demo@parkvision.com / ParkVision123');
  }
});
