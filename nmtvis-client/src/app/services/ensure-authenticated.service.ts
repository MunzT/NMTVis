import {Injectable} from '@angular/core';
import {AuthService} from './auth.service';
import {CanActivate, Router} from '@angular/router';

@Injectable()
export class EnsureAuthenticated implements CanActivate {
    constructor(private auth: AuthService, private router: Router) {
    }

    canActivate(): boolean {
        if (localStorage.getItem('access_token')) {
            return true;
        }
        else {
            this.router.navigate(['/login']);
            return false;
        }
    }
}