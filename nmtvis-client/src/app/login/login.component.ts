import {Component, OnInit} from '@angular/core';
import {Router} from '@angular/router';
import {AuthService} from '../services/auth.service';

@Component({
    selector: 'app-login',
    templateUrl: './login.component.html',
    styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {

    constructor(private router: Router, readonly auth: AuthService) {
    }

    username: string;
    password: string;
    error: string;

    login() {
        this.auth.login({password: this.password, username: this.username})
            .subscribe(result => {
                localStorage.setItem('access_token', result.access_token);
                localStorage.setItem('username', result.username);
                this.router.navigate(['/documents']);
            }, error => {
                this.error = "Login failed";
            });
    }

    ngOnInit() {
    }

}
