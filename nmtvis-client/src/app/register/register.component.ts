import {Component, OnInit} from '@angular/core';
import {Router} from '@angular/router';
import {AuthService} from '../services/auth.service';

@Component({
    selector: 'app-register',
    templateUrl: './register.component.html',
    styleUrls: ['./register.component.css']
})
export class RegisterComponent implements OnInit {

    username: string;
    password: string;
    error: string;

    constructor(private router: Router, readonly auth: AuthService) {
    }

    register() {
        this.auth.register({password: this.password, username: this.username})
            .subscribe(result => {
                localStorage.setItem('access_token', result.access_token);
                localStorage.setItem('username', result.username);
                this.router.navigate(['/intro']);
            }, error => {
                this.error = error.error.msg;
            });
    }

    ngOnInit() {
    }

}
