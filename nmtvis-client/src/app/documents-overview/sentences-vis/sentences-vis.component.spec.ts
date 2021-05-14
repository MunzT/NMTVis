import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SentencesVisComponent } from './sentences-vis.component';

describe('SentencesVisComponent', () => {
  let component: SentencesVisComponent;
  let fixture: ComponentFixture<SentencesVisComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SentencesVisComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SentencesVisComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
