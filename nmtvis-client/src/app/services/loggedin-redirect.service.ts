import {Injectable} from '@angular/core';
import {CanActivate, Router} from '@angular/router';
import {AuthService} from './auth.service';

@Injectable()
export class LoggedinRedirect implements CanActivate {
    constructor(private auth: AuthService, private router: Router) {
    }

    canActivate(): boolean {
        if (localStorage.getItem('access_token')) {
            this.router.navigate(['/documents']);
            return false;
        }
        else {
            return true;
        }
    }
}
