import 'rxjs/add/operator/do';
import {Observable} from 'rxjs/Observable';
import {Injectable} from '@angular/core';
import {
    HttpResponse,
    HttpErrorResponse,
    HttpRequest,
    HttpHandler,
    HttpEvent,
    HttpInterceptor
} from '@angular/common/http';
import {AuthService} from '../services/auth.service';
import {Router} from '@angular/router';

@Injectable()
export class JwtInterceptor implements HttpInterceptor {
    constructor(public auth: AuthService, private router: Router) {
    }

    intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {

        return next.handle(request).do((event: HttpEvent<any>) => {
            if (event instanceof HttpResponse) {
                // do stuff with response if you want
            }
        }, (err: any) => {
            if (err instanceof HttpErrorResponse) {
                if (err.status === 401 || err.status === 422) {
                    // redirect to the login route
                    // or show a modal
                    this.router.navigate(['/login']);
                }
            }
        });
    }
}