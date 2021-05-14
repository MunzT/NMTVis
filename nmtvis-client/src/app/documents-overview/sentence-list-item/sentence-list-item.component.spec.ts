import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SentenceListItemComponent } from './sentence-list-item.component';

describe('SentenceListItemComponent', () => {
  let component: SentenceListItemComponent;
  let fixture: ComponentFixture<SentenceListItemComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SentenceListItemComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SentenceListItemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
