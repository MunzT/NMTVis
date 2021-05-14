import { TestBed, inject } from '@angular/core/testing';

import { LoggedinRedirectService } from './loggedin-redirect.service';

describe('LoggedinRedirectService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [LoggedinRedirectService]
    });
  });

  it('should be created', inject([LoggedinRedirectService], (service: LoggedinRedirectService) => {
    expect(service).toBeTruthy();
  }));
});
