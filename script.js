const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,method='GET',body){
 const r=await fetch(url,{method,headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
 const d=await r.json().catch(()=>({ok:false,error:'Bad response'}));
 if(r.status===401&&!/login|register|index/.test(location.pathname)&&location.pathname!=='/')location.href='login.html';
 if(!d.ok)throw new Error(d.error||'Request failed');return d;}
async function nav(authed){
 const h=document.getElementById('nav');
 h.innerHTML='<a class="logo" href="index.html">ParkVision</a>'+(authed?
  '<a href="dashboard.html">Dashboard</a><a href="parking.html">Parking</a><a href="vehicles.html">Vehicles</a><a href="#" id="out">Log out</a>':
  '<a href="login.html">Login</a><a href="register.html">Register</a>');
 const o=document.getElementById('out');if(o)o.onclick=async e=>{e.preventDefault();await api('/api/auth/logout','POST');location.href='index.html';};}
async function requireAuth(){try{const d=await api('/api/auth/me');await nav(true);return d.user;}catch{location.href='login.html';}}
function live(fn){if(window.io){io().on('slots:changed',fn);}setInterval(fn,30000);}
function fmt(d){return new Date(d).toLocaleString();}
