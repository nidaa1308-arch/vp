require('dotenv').config();const m=require('mysql2/promise');
(async()=>{const e=process.argv[2];if(!e)return console.log('usage: npm run make-admin -- email');
const c=await m.createConnection({host:process.env.DB_HOST,port:process.env.DB_PORT,user:process.env.DB_USER,password:process.env.DB_PASSWORD,database:process.env.DB_NAME});
const [r]=await c.execute("UPDATE users SET role='admin' WHERE email=?",[e]);console.log(r.affectedRows?'Promoted '+e:'No such user');await c.end();})();
