import os, tempfile
from pathlib import Path

# Isolate test data before importing the app.
_tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
_tmp.close()
os.environ['SAKHI_DB'] = _tmp.name
os.environ['SAKHI_SECRET'] = 'test-secret-not-for-production'
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health():
    r=client.get('/api/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'

def test_register_journal_analytics_and_delete():
    r=client.post('/api/auth/register',json={'name':'Test User','email':'test@example.com','password':'StrongPass123'})
    assert r.status_code == 200
    token=r.json()['token']; headers={'Authorization':'Bearer '+token}
    r=client.post('/api/journal',headers=headers,json={'entry_date':'2026-11-20','mood':'Calm','symptoms':['Cramps'],'notes':'Test note'})
    assert r.status_code == 200
    entry=r.json(); assert entry['symptoms']==['Cramps']
    assert client.get('/api/analytics',headers=headers).json()['total_entries']==1
    assert client.delete(f"/api/journal/{entry['id']}",headers=headers).status_code==200
    assert client.get('/api/journal',headers=headers).json()==[]

def test_login_and_assistant():
    client.post('/api/auth/register',json={'name':'Ask User','email':'ask@example.com','password':'StrongPass123'})
    r=client.post('/api/auth/login',json={'email':'ask@example.com','password':'StrongPass123'})
    assert r.status_code==200
    headers={'Authorization':'Bearer '+r.json()['token']}
    answer=client.post('/api/ask',headers=headers,json={'question':'What is menstrual health?'})
    assert answer.status_code==200 and 'answer' in answer.json()
