postgresql://postgres:[SAU@2003mya]@db.inzjubyfhnabmcntvozq.supabase.co:5432/postgres

{
  "name": "automotto-api",
  "version": "1.0.0",
  "description": "AUTOMOTTO API for mobile app",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "pg": "^8.11.3",
    "jsonwebtoken": "^9.0.2",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1",
    "helmet": "^7.1.0",
    "express-rate-limit": "^7.1.5"
  },
  "devDependencies": {
    "nodemon": "^3.0.2"
  }
}

{
  "name": "automotto-api",
  "version": "1.0.0",
  "description": "AUTOMOTTO API for mobile app",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "pg": "^8.11.3",
    "jsonwebtoken": "^9.0.2",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1",
    "helmet": "^7.1.0",
    "express-rate-limit": "^7.1.5"
  },
  "devDependencies": {
    "nodemon": "^3.0.2"
  }
}
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

const { Pool } = require('pg');

const app = express();
const PORT = process.env.PORT || 3000;

// Database connection
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: {
    rejectUnauthorized: false
  }
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Health check
app.get('/health', async (req, res) => {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT NOW()');
    client.release();
    res.json({ status: 'ok', timestamp: result.rows[0].now });
  } catch (err) {
    res.status(500).json({ status: 'error', message: err.message });
  }
});

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    res.status(403).json({ error: 'Invalid token' });
  }
};

// Vehicles routes
app.get('/api/vehicles', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(
      'SELECT id, make, model, year, nickname, created_at FROM vehicles WHERE user_id = $1',
      [req.user.id]
    );
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching vehicles:', err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/vehicles/:id', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(
      'SELECT v.*, u.name as owner_name FROM vehicles v JOIN users u ON v.user_id = u.id WHERE v.id = $1 AND u.id = $2',
      [req.params.id, req.user.id]
    );
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Vehicle not found' });
    }
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Trips routes
app.get('/api/trips', authenticateToken, async (req, res) => {
  const { vehicle_id, limit = 10, start_date, end_date } = req.query;
  
  let query = 'SELECT id, vehicle_id, started_at, ended_at, distance_km, avg_speed_kph, status FROM trips WHERE vehicle_id IN (SELECT id FROM vehicles WHERE user_id = $1)';
  let params = [req.user.id];
  let paramIndex = 2;

  if (vehicle_id) {
    query += ' AND vehicle_id = $' + paramIndex++;
    params.push(vehicle_id);
  }
  
  if (start_date) {
    query += ' AND started_at >= $' + paramIndex++;
    params.push(start_date);
  }
  
  if (end_date) {
    query += ' AND ended_at <= $' + paramIndex++;
    params.push(end_date);
  }

  query += ' ORDER BY started_at DESC LIMIT $' + paramIndex++;
  params.push(parseInt(limit));

  try {
    const result = await pool.query(query, params);
    res.json(result.rows);
  } catch (err) {
    console.error('Error fetching trips:', err);
    res.status(500).json({ error: err.message });
  }
});

// Driver scores
app.get('/api/driver-scores', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT ds.*, t.distance_km, t.avg_speed_kph 
      FROM driver_scores ds
      JOIN trips t ON ds.trip_id = t.id
      JOIN vehicles v ON t.vehicle_id = v.id
      WHERE v.user_id = $1
      ORDER BY ds.computed_at DESC
      LIMIT 20
    `, [req.user.id]);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Health snapshots
app.get('/api/health', authenticateToken, async (req, res) => {
  const { vehicle_id } = req.query;
  
  try {
    const result = await pool.query(
      'SELECT * FROM health_snapshots WHERE vehicle_id = $1 ORDER BY ts DESC LIMIT 10',
      [vehicle_id]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// SOS events
app.post('/api/sos', authenticateToken, async (req, res) => {
  const { vehicle_id, lat, lng, distress_phrase } = req.body;
  
  try {
    const result = await pool.query(
      `INSERT INTO sos_events (id, vehicle_id, user_id, triggered_by, distress_phrase, detected_ts, lat, lng, status)
       VALUES (gen_random_uuid(), $1, $2, 'manual', $3, NOW(), $4, $5, 'pending')
       RETURNING id`,
      [vehicle_id, req.user.id, distress_phrase || 'Mayday Mayday', lat, lng]
    );
    
    // Trigger notification (simplified)
    await pool.query(
      'INSERT INTO notifications (id, user_id, channel, template_key, title, body, status, created_at)
       VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, NOW())',
      [req.user.id, 'push', 'sos_alert', 'SOS Alert Activated', 'Emergency alert sent to your contacts', 'queued']
    );
    
    res.json({ 
      success: true, 
      sos_id: result.rows[0].id,
      message: 'SOS alert triggered successfully' 
    });
  } catch (err) {
    console.error('Error triggering SOS:', err);
    res.status(500).json({ error: err.message });
  }
});

// Documents
app.get('/api/documents', authenticateToken, async (req, res) => {
  const { vehicle_id } = req.query;
  
  try {
    let query = `
      SELECT d.*, v.nickname as vehicle_name 
      FROM documents d
      LEFT JOIN vehicles v ON d.vehicle_id = v.id
      WHERE d.user_id = $1
    `;
    let params = [req.user.id];
    
    if (vehicle_id) {
      query += ' AND d.vehicle_id = $2';
      params.push(vehicle_id);
    }
    
    const result = await pool.query(query, params);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
  console.log(`AUTOMOTTO API running on port ${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, closing database connections...');
  pool.end();
  process.exit(0);
});
