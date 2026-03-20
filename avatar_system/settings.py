import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-dev-key-replace-in-production')
DEBUG      = os.environ.get('DEBUG', '1') == '1'
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'channels',
    'core',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'avatar_system.middleware.LoginRequiredMiddleware',
]

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/login/'

ROOT_URLCONF = 'avatar_system.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.debug',
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
    ]},
}]

ASGI_APPLICATION = 'avatar_system.asgi.application'
WSGI_APPLICATION = 'avatar_system.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL    = '/static/'
STATIC_ROOT   = os.environ.get('STATIC_ROOT',  str(BASE_DIR / 'staticfiles'))
STATICFILES_DIRS = [BASE_DIR / 'static'] if (BASE_DIR / 'static').exists() else []

MEDIA_URL     = '/media/'
MEDIA_ROOT    = os.environ.get('MEDIA_ROOT',   str(BASE_DIR / 'media'))

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Avatar storage – override via environment
AVATAR_DATA_ROOT = os.environ.get('AVATAR_DATA_ROOT', str(BASE_DIR / 'avatar_data'))
VIDEO_SCAN_ROOT  = os.environ.get('VIDEO_SCAN_ROOT',  str(BASE_DIR / 'video_data'))

# Pre-trained model weights
# Expected layout under this directory:
#   smplx/SMPLX_NEUTRAL.npz   (and SMPLX_MALE.npz, SMPLX_FEMALE.npz)
#   deca/                      (DECA/FLAME weights – see /opt/deca in Docker)
#   realesrgan/                (auto-downloaded on first use)
SMPLX_MODEL_DIR = os.environ.get('SMPLX_MODEL_DIR', str(BASE_DIR / 'models'))

# Channels – Redis in production, InMemory for local dev
REDIS_URL = os.environ.get('REDIS_URL', '')

if REDIS_URL:
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels_redis.core.RedisChannelLayer',
            'CONFIG':  {'hosts': [REDIS_URL]},
        }
    }
else:
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels.layers.InMemoryChannelLayer',
        }
    }
