from django.shortcuts import redirect
from django.conf import settings


class LoginRequiredMiddleware:
    EXEMPT_PREFIXES = ('/login/', '/logout/', '/admin/', '/static/', '/media/')

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.user.is_authenticated:
            if not any(request.path.startswith(p) for p in self.EXEMPT_PREFIXES):
                return redirect(f'{settings.LOGIN_URL}?next={request.path}')
        return self.get_response(request)
