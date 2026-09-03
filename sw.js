const CACHE='nfl-pools-2026-build22';
const APP_SHELL=[
  './',
  './index.html',
  './manifest.json',
  './odds.json',
  'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2'
];

self.addEventListener('install',event=>{
  event.waitUntil((async()=>{
    const cache=await caches.open(CACHE);
    await Promise.allSettled(APP_SHELL.map(async url=>{
      try{
        const req=new Request(url,{cache:'reload',mode:url.startsWith('http')?'no-cors':'same-origin'});
        const res=await fetch(req);
        if(res && (res.ok || res.type==='opaque')) await cache.put(url,res.clone());
      }catch(_){}
    }));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate',event=>{
  event.waitUntil((async()=>{
    const keys=await caches.keys();
    await Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch',event=>{
  const req=event.request;
  if(req.method!=='GET')return;
  const url=new URL(req.url);

  // App navigation: network first, cached app fallback.
  if(req.mode==='navigate'){
    event.respondWith((async()=>{
      try{
        const fresh=await fetch(req);
        const cache=await caches.open(CACHE);
        cache.put('./index.html',fresh.clone());
        cache.put('./',fresh.clone());
        return fresh;
      }catch(_){
        return (await caches.match('./index.html')) || (await caches.match('./'));
      }
    })());
    return;
  }

  // Odds: prefer fresh, use last cached copy when offline.
  if(url.pathname.endsWith('/odds.json') || url.hostname==='raw.githubusercontent.com'){
    event.respondWith((async()=>{
      const cache=await caches.open(CACHE);
      try{
        const fresh=await fetch(req);
        if(fresh && fresh.ok) await cache.put(req,fresh.clone());
        return fresh;
      }catch(_){
        const exact=await cache.match(req);
        if(exact)return exact;
        const sameOrigin=await cache.match('./odds.json');
        if(sameOrigin)return sameOrigin;
        return new Response('{"games":{}}',{headers:{'Content-Type':'application/json'}});
      }
    })());
    return;
  }

  // Supabase library and same-origin static resources: cache first, then network.
  if(url.hostname==='cdn.jsdelivr.net' || url.origin===self.location.origin){
    event.respondWith((async()=>{
      const cached=await caches.match(req);
      if(cached)return cached;
      try{
        const fresh=await fetch(req);
        const cache=await caches.open(CACHE);
        if(fresh && (fresh.ok || fresh.type==='opaque')) await cache.put(req,fresh.clone());
        return fresh;
      }catch(_){
        return cached || Response.error();
      }
    })());
  }
});
